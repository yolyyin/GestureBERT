import os
from torch.utils.data import Dataset
import pickle
import torch
import random
from utils import smpl_to_openpose,openpose_to_smpl

VALID_JOINTS=list(range(22))+list(range(25,55))+list(range(66,76)) # smplx body and hand joints
PAD_IDX= -3.0
MASK_IDX= 2.0
UNK_IDX = -2.0
SOS_IDX=3.0



class BERTDataset(Dataset):
    def __init__(self,data_file_dir,seq_len,n_frame_file,full_seq=False):
        super(BERTDataset, self).__init__()
        self.full_seq=full_seq
        self.seq_len= seq_len
        self.n_frame_per_file = n_frame_file
        self.n_files= len(os.listdir(data_file_dir))
        self.data_fns = [os.path.join(data_file_dir, fn) for fn in sorted(os.listdir(data_file_dir))]

        # prepare file list to get full sequence gestures
        video_names=[os.path.basename(fn).split('.')[0] for fn in sorted(os.listdir(data_file_dir))]
        frame_infos=[os.path.basename(fn).split('.')[1] for fn in sorted(os.listdir(data_file_dir))]
        self.full_seq_data_fns=[]
        t_num_file=0
        valid_file_per_clip=[]
        cur_video_name=video_names[0]
        for i,(vn,frame_info,data_fn) in enumerate(zip(video_names,frame_infos,self.data_fns)):
            start_frame=int(frame_info.split('_')[0])
            end_frame=int(frame_info.split('_')[1])
            if vn==cur_video_name and (end_frame-start_frame+1==self.n_frame_per_file):
                # only account for files with full n_frame_per_file length
                t_num_file += 1
                valid_file_per_clip.append(data_fn)
            elif vn!=cur_video_name and (end_frame-start_frame+1==self.n_frame_per_file):
                # discard extra files whose frame numbers is lesser than self.seq_len
                t_num_file = t_num_file - ((t_num_file*self.n_frame_per_file) % self.seq_len)//self.n_frame_per_file
                valid_file_per_clip = valid_file_per_clip[0:t_num_file]
                self.full_seq_data_fns.extend(valid_file_per_clip)
                # count for the next clip
                t_num_file = 1
                valid_file_per_clip.clear()
                valid_file_per_clip.append(data_fn)
                cur_video_name = vn
            else:
                # discard files whose frame numbers is lesser than self.seq_len
                continue
        print(len(self.full_seq_data_fns))
        print()

        # openpose to smplx joint map to permute openpose joint masks
        smplx2op=smpl_to_openpose(model_type='smplx',use_hands=True,
                                         use_face=True, use_face_contour=True, openpose_format='coco25')
        self.op2smplx = torch.tensor(openpose_to_smpl(smplx2op))

        sample_data_file=self.data_fns[0]
        with open(sample_data_file, "rb") as f:
            data = pickle.load(f) # dict
            joint_angles = torch.tensor(data['normal_joints']) # shape [T,127,3]
            joint_masks = torch.tensor(data['joint_masks']) # shape [T,121]
            ges_labels = torch.tensor(data['ges_labels']) # shape [T]
        assert joint_angles.shape[0] == joint_masks.shape[0] and \
               joint_angles.shape[0] == ges_labels.shape[0], "joint angle tensor, mask tensor and gest labels must have same number of frames!"

        if self.full_seq:
            self.data_len = (len(self.full_seq_data_fns) * self.n_frame_per_file) // self.seq_len
        else:
            self.data_len = (self.n_files * self.n_frame_per_file) // self.seq_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        n_file = self.seq_len // self.n_frame_per_file
        file_idx = item * self.seq_len // self.n_frame_per_file
        if self.full_seq:
            fns = self.full_seq_data_fns[file_idx:file_idx + n_file]
        else:
            fns = self.data_fns[file_idx:file_idx+n_file]
        # abort files that does not belong to the same video
        video_keyword = os.path.basename(fns[0]).split('.')[0]
        fns = [file for file in fns if os.path.basename(file).split('.')[0]==video_keyword]
        joint_angles_list=[]
        lhand_list=[]
        rhand_list=[]
        vit_vec_list=[]
        joint_masks_list=[]
        ges_labels_list=[]
        for file in fns:
            with open(file, "rb") as f:
                data = pickle.load(f)  # dict
                joint_angles = torch.tensor(data['joint_angles'])  # shape [n_frame,127,3]
                lhand_ncs = torch.tensor(data['lhand_norms'])  # shape [T,6]
                rhand_ncs = torch.tensor(data['rhand_norms'])  # shape [T,6]
                vit_vecs = torch.tensor(data['hand_shape_vecs'])  # shape [T,256]
                joint_masks = torch.tensor(data['joint_masks'],dtype=torch.long)  # shape [n_frame,118]
                ges_labels = torch.tensor(data['ges_labels'],dtype=torch.long) # shape [n_frame,]
                #print(file)
                #print(joint_angles.shape)
                #print(lhand_ncs.shape)
                #print(rhand_ncs.shape)
                #print(vit_vecs.shape)
                #print(joint_masks.shape)
                #print(ges_labels.shape)
                joint_angles_list.append(joint_angles)
                lhand_list.append(lhand_ncs)
                rhand_list.append(rhand_ncs)
                vit_vec_list.append(vit_vecs)
                joint_masks_list.append(joint_masks)
                ges_labels_list.append(ges_labels)
        joint_angles = torch.concatenate(joint_angles_list) # shape [T,127,3]
        lhand_ncs=torch.concatenate(lhand_list) # shape [T,6]
        rhand_ncs=torch.concatenate(rhand_list) # shape [T,6]
        vit_vecs=torch.concatenate(vit_vec_list) # shape [T,256]
        op_joint_masks = torch.concatenate(joint_masks_list) # shape [T,118]
        #open_pose to smplx mask sequence
        joint_masks = torch.index_select(op_joint_masks,1,self.op2smplx[VALID_JOINTS]) # [T,62]
        ges_labels=torch.concatenate(ges_labels_list) # shape [T,]

        joint_angles = joint_angles[:,VALID_JOINTS,:] # shape [T,62,3]

        body_masks = joint_masks[:,:,None] #[T,62,1]
        masked_joint_angles = body_masks * joint_angles + (1 - body_masks) * torch.ones_like(
            joint_angles) * UNK_IDX  # shape [T,62,3]

        # judge whether either hand is visible from open pose joint masks
        # if for either hand visible joints number<3, then the norm and center vector for this hand
        # is set to be UNK
        lh_joints_mask=op_joint_masks[:,25:46] # shape [T,21]
        lh_mask=(lh_joints_mask.sum(dim=-1,keepdim=True) > 2).to(torch.long) #[T,1]
        lhand_ncs = lhand_ncs*lh_mask + (1 - lh_mask) * torch.ones_like(lhand_ncs) * UNK_IDX # [T,6]
        rh_joints_mask=op_joint_masks[:,46:67] # shape [T,21]
        rh_mask=(rh_joints_mask.sum(dim=-1,keepdim=True) > 2).to(torch.long) #[T,1]
        rhand_ncs = rhand_ncs * rh_mask + (1 - rh_mask) * torch.ones_like(rhand_ncs) * UNK_IDX  # [T,6]

        t= masked_joint_angles.shape[0]
        sequence = torch.concatenate([masked_joint_angles.view(t,-1),
                                      lhand_ncs,rhand_ncs,vit_vecs],dim=-1) # [t,454]
        (t,d)=sequence.shape
        n_joints=body_masks.shape[1]

        seq_random,seq_label=self.random_frame(sequence) # [t,454]
        sos_padding = torch.ones(1,d,dtype=torch.float32)*SOS_IDX #[1,454]
        sos_label_padding=torch.ones(1,d,dtype=torch.float32)*PAD_IDX #[1,454]

        paddings = [torch.ones(d,dtype=torch.float32)*PAD_IDX for _ in range(self.seq_len-seq_random.shape[0])]
        pose_paddings_prev= torch.zeros(1,n_joints,1,dtype=body_masks.dtype)
        pose_paddings_after = [torch.zeros(n_joints,1,dtype=body_masks.dtype) for _ in range(self.seq_len-seq_random.shape[0])]
        if len(paddings)>0:
            paddings = torch.stack(paddings,dim=0)
            pose_paddings_after = torch.stack(pose_paddings_after,dim=0)  # [x,62,1]
            # add paddings and flatten the input sequence and the labels to the shape [seq_len,pose_dim]
            bert_input = torch.concatenate([sos_padding,seq_random,paddings],dim=0).view(self.seq_len+1,-1) # [seq_len+1,454]
            bert_label = torch.concatenate([sos_label_padding,seq_label,paddings],dim=0).view(self.seq_len+1,-1)  # [seq_len+1,454]
            bert_pose_mask = torch.concatenate([pose_paddings_prev,body_masks,pose_paddings_after],dim=0).view(self.seq_len+1,-1)  # [seq_len+1,62]

            ges_input = torch.concatenate([sos_padding,sequence,paddings],dim=0).view(self.seq_len+1,-1) # [seq_len+1,454]
        else:
            bert_input = torch.concatenate([sos_padding,seq_random],dim=0).view(self.seq_len+1,-1) # [seq_len+1,454]
            bert_label = torch.concatenate([sos_label_padding,seq_label],dim=0).view(self.seq_len+1,-1)  # [seq_len+1,454]
            bert_pose_mask = torch.concatenate([pose_paddings_prev,body_masks],dim=0).view(self.seq_len+1,-1)  # [seq_len+1,62]

            ges_input = torch.concatenate([sos_padding,sequence],dim=0).view(self.seq_len+1,-1) # [seq_len+1,454]

        ges_paddings_prev = torch.zeros(1, dtype=ges_labels.dtype)
        ges_paddings_after = torch.zeros(self.seq_len - ges_labels.shape[0],dtype=ges_labels.dtype)
        if ges_paddings_after.shape[0]>0:
            ges_label = torch.concatenate([ges_paddings_prev,ges_labels, ges_paddings_after], dim=0)  # [seq_len+1]
        else:
            ges_label = torch.concatenate([ges_paddings_prev,ges_labels],dim=0)  # [seq_len+1]


        output ={"bert_input": bert_input,
                 "bert_label": bert_label,
                 "bert_pose_mask": bert_pose_mask,
                 "ges_input": ges_input,
                 "ges_label": ges_label,#}
                 "img_paths":fns}
        return output

    @staticmethod
    def random_frame(sequence):
        output_labels=[]
        (t,d) = sequence.shape
        # forcefully replace 1~4 frames at the end of the sequence,
        # so as to give the model some predictability ability
        num_end_frame=random.choice([1,2,3,4])
        for i in range(t):
            prob = random.random()
            actual_frame = sequence[i, :].clone()
            if i>=t-num_end_frame or prob <0.2:#0.15:
                prob /= 0.2 #0.15
                # 80% randomly change frame to mask frame->85%
                if i>=t-num_end_frame or prob<0.8:
                    sequence[i,:]=torch.ones(d,dtype=torch.float32)*MASK_IDX
                # 10% randomly change frame to random frame
                elif prob<0.9:
                    sequence[i, :] = torch.rand(d)*2.0-1.0 # range[-1,1]
                # 10% keep frame to actual frame->5%
                # append the target frame(actual frame)
                output_labels.append(actual_frame)
            else:
                # unchanged frames don't need to be predicted
                #output_labels.append(torch.ones(d,dtype=torch.float32)*PAD_IDX)
                output_labels.append(actual_frame)
        output_labels = torch.stack(output_labels,dim=0)
        return sequence,output_labels

    def get_full_seq_fns(self):
        return self.full_seq_data_fns



