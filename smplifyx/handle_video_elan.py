import time
import os
import pympi
import pickle

import numpy as np
from ultralytics import YOLO
from palm_orientation import get_origin_hand_norm
from hand_cropper import HandEncoder,get_vit_vec


if __name__=="__main__":
    vit_model = HandEncoder(vit_pth_path='E:/data/gestureDataset/VitB16.pth').to('cuda')
    yolo_model = YOLO("E:/data/gestureDataset/YOLOv10x_hands.pt").to('cuda')
    with open('E:/data/smplx_multisimo/vit_pca.pkl','rb') as f:
        pca_model = pickle.load(f)
    n_frame_per_file=16
    FPS=30
    buffer_seconds=5
    origin_video_names=[os.path.splitext(fn)[0] for fn in os.listdir("E:/data/smplx_multisimo/original_videos")]
    img_cache_folder="E:/data/smplx_multisimo/caches"
    elan_anno_dir="E:/data/smplx_multisimo/eafs"
    input_folder = 'E:/data/smplx_multisimo/smplx_poses'
    out_folder = 'E:/data/smplx_multisimo/smplx_w_gesgroup_hnorm_hshape128'
    out_ges_pos_folder='E:/data/smplx_multisimo/smplx_GesPos_w_hnorm_hshape128'
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(out_ges_pos_folder,exist_ok=True)

    clip2ges={}
    start=time.time()
    for video_keyword in sorted(origin_video_names):
        ids = video_keyword.split('_')
        person_id,session_id= "",""
        for id in ids:
            if id.startswith("S"):
                session_id = id
            else:
                person_id = id
        eaf_path = os.path.join(elan_anno_dir, f"{session_id}.eaf")
        tier_name = f"Gestures_{person_id}_{session_id}" if person_id.startswith("M") \
            else f"Gestures_{person_id}"
        try:
            # Load the ELAN file
            eaf = pympi.Elan.Eaf(eaf_path)

            if tier_name not in eaf.tiers:
                raise ValueError(f"Tier '{tier_name}' not found in the ELAN file.")

            # Get annotations in the tier
            annotations = eaf.get_annotation_data_for_tier(tier_name)

            for idx, (start_ms, end_ms, _) in enumerate(annotations):
                clip_keyword=f"{video_keyword}_clip_{idx:04d}"

                duration_ms = (end_ms-start_ms+2*buffer_seconds*1000.0)
                num_frames = int(np.ceil(duration_ms * FPS / 1000.0))
                gesture_array = np.zeros(num_frames, dtype=np.int64)

                start_frame = int(np.floor(buffer_seconds * FPS))
                end_frame = int(np.ceil(((end_ms-start_ms)/1000.0+buffer_seconds) * FPS))
                gesture_array[start_frame:end_frame] = 1

                # save gesture_array of every clip
                clip2ges[clip_keyword]=gesture_array
        except  Exception as e:
            print(f"Can't extract ges group info from file {eaf_path} with tier {tier_name}", e)
            exit(1)






    pose_embedding_per_clip=[]
    joint_per_clip=[]
    joint_mask_per_clip=[]
    ges_label_per_clip=[]
    norm_joint_per_clip=[]
    lhand_norm_per_clip=[]
    rhand_norm_per_clip=[]
    vit_vec_per_clip=[]
    cur_clip_keyword=""
    total_pos_frame=0
    # read saved pkl file to determine which frames to save
    for input_fn in sorted(os.listdir(input_folder)):
        input_path = os.path.join(input_folder,input_fn)
        ouput_path = os.path.join(out_folder,input_fn)

        input_info=input_fn.split('.')
        clip_keyword = input_info[0]
        if (clip_keyword != cur_clip_keyword) :
            if cur_clip_keyword =="":
                cur_clip_keyword = clip_keyword
            else:
                # concat all frames in a clip
                pose_embeds= np.concatenate(pose_embedding_per_clip,axis=0)
                joints=np.concatenate(joint_per_clip,axis=0)
                jmasks=np.concatenate(joint_mask_per_clip,axis=0)
                glabels=np.concatenate(ges_label_per_clip,axis=0)
                norm_joints=np.concatenate(norm_joint_per_clip,axis=0)
                lh_norms=np.concatenate(lhand_norm_per_clip,axis=0)
                rh_norms=np.concatenate(rhand_norm_per_clip,axis=0)
                vits=np.concatenate(vit_vec_per_clip,axis=0)
                # judge where are ges positive frames and delete other frames
                ges_pos_idx=np.where(glabels==1)[0]
                pose_embeds=pose_embeds[ges_pos_idx]
                joints=joints[ges_pos_idx]
                jmasks=jmasks[ges_pos_idx]
                glabels=glabels[ges_pos_idx]
                norm_joints=norm_joints[ges_pos_idx]
                lh_norms=lh_norms[ges_pos_idx]
                rh_norms=rh_norms[ges_pos_idx]
                vits=vits[ges_pos_idx]
                # save them in out_ges_pos_folder according to n_frame_per_file
                nframes=len(ges_pos_idx)
                total_pos_frame+=nframes
                print(f"\nNumber of Positive Frames for {cur_clip_keyword}:{nframes}")
                print()
                for i in range(0,nframes,n_frame_per_file):
                    end_frame=min(i+n_frame_per_file,nframes)
                    frame_keyword=f"{i+1:06d}_{end_frame:06d}"
                    pos_save_dict={'pose_embeddings':pose_embeds[i:end_frame],
                          'joint_angles':joints[i:end_frame],
                          'joint_masks':jmasks[i:end_frame],
                          'ges_labels':glabels[i:end_frame],
                          'normal_joints':norm_joints[i:end_frame],
                          'lhand_norms':lh_norms[i:end_frame],
                          'rhand_norms':rh_norms[i:end_frame],
                          'hand_shape_vecs':vits[i:end_frame]}
                    save_name = f"{cur_clip_keyword}.{frame_keyword}.pkl"
                    save_fn = os.path.join(out_ges_pos_folder, save_name)
                    with open(save_fn,'wb') as f:
                        pickle.dump(pos_save_dict,f,protocol=2)
                    print(f"Positive Ges Data Saved!:{save_fn}")
                cur_clip_keyword = clip_keyword
                pose_embedding_per_clip.clear()
                joint_per_clip.clear()
                joint_mask_per_clip.clear()
                ges_label_per_clip.clear()
                norm_joint_per_clip.clear()
                lhand_norm_per_clip.clear()
                rhand_norm_per_clip.clear()
                vit_vec_per_clip.clear()

        clip_ges_array = clip2ges[clip_keyword]

        start_frame = int(input_info[1].split('_')[0])-1
        end_frame = min(int(input_info[1].split('_')[1]),len(clip_ges_array))
        if end_frame<=start_frame:
            print()

        ges_array = clip_ges_array[start_frame:end_frame] # shape [T]
        with open(input_path, "rb") as input_file:
            data = pickle.load(input_file)  # dict
        joint_list=[]
        lnorm_list=[]
        rnorm_list=[]
        vit_vec_list=[]
        for frame_idx in range(start_frame,end_frame):
            img_name= f"frame_{frame_idx:05d}.jpg"
            img_path = os.path.join(img_cache_folder,clip_keyword,'images',img_name)

            smplx_joints = data['joint_angles'][frame_idx-start_frame] #[127,3]
            hand_ori_result = get_origin_hand_norm(smplx_joints)
            lhand_n=np.concatenate((hand_ori_result["lhand_norm"],hand_ori_result["lhand_cen"])) #[6,]
            rhand_n=np.concatenate((hand_ori_result['rhand_norm'], hand_ori_result['rhand_cen']))#[6,]
            origin = hand_ori_result['origin'] #[3,]
            new_joints=smplx_joints-origin[None,:] #[127,3]
            vit_vec_l = get_vit_vec(img_path,yolo_model,vit_model,'cuda')#[768]
            if len(vit_vec_l)==0:
                # no hand, set shape vec to 0
                vit_vec = np.zeros((256,), dtype=np.float32)  # [256]
            elif len(vit_vec_l)==1:
                # one hand, pad shape vec to 256
                vit_vec= vit_vec_l[0]
                vit_vec = np.squeeze(pca_model.transform(vit_vec[np.newaxis, ...]))  # [128]
                vit_vec=np.concatenate((vit_vec,np.zeros((128,),dtype=np.float32)),axis=0)  # [256]
            else:
                vit_vec1,vit_vec2 = vit_vec_l[0],vit_vec_l[1]
                vit_vec1 = np.squeeze(pca_model.transform(vit_vec1[np.newaxis, ...]))
                vit_vec2 = np.squeeze(pca_model.transform(vit_vec2[np.newaxis, ...]))
                vit_vec=np.concatenate((vit_vec1,vit_vec2),axis=0)

            joint_list.append(new_joints)
            lnorm_list.append(lhand_n)
            rnorm_list.append(rhand_n)
            vit_vec_list.append(vit_vec)
        joints=np.stack(joint_list,axis=0)  # [T,127,3]
        lnorms=np.stack(lnorm_list,axis=0)  # [T,6]
        rnorms=np.stack(rnorm_list,axis=0)  # [T,6]
        vit_vecs=np.stack(vit_vec_list,axis=0)  # [T,256]

        with open(ouput_path, 'wb') as output_file:
            data['ges_labels']=ges_array
            data['normal_joints']=joints
            data['lhand_norms']=lnorms
            data['rhand_norms']=rnorms
            data['hand_shape_vecs']=vit_vecs

            pickle.dump(data, output_file, protocol=2)
        print(f"Done saving ges group info to '{ouput_path}'")

        pose_embedding_per_clip.append(data['pose_embeddings'])
        joint_per_clip.append(data['joint_angles'])
        joint_mask_per_clip.append(data['joint_masks'])
        ges_label_per_clip.append(ges_array)
        norm_joint_per_clip.append(joints)
        lhand_norm_per_clip.append(lnorms)
        rhand_norm_per_clip.append(rnorms)
        vit_vec_per_clip.append(vit_vecs)

    end = time.time()
    print(f"Gesture group info preprocess completes. Time spent:{end - start:.4f} seconds")
    print(f"Total Positive Frame Numbers: {total_pos_frame}")
