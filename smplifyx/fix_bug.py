import os
import pickle

if __name__ == '__main__':
    input_folder="/home/ubuntu/Documents/data/smplx_multisimo/new_smplx_poses"
    output_folder="/home/ubuntu/Documents/data/smplx_multisimo/new_smplx_poses_fixed"
    os.makedirs(output_folder, exist_ok=True)
    n_frame_per_file=16
    for input_fn in sorted(os.listdir(input_folder)):
        input_path=os.path.join(input_folder,input_fn)
        output_path=os.path.join(output_folder,input_fn)
        input_info=input_fn.split('.')
        clip_keyword=input_info[0]
        frame_keyword=input_info[1]
        start_frame=int(frame_keyword.split("_")[0])-1
        end_frame=int(frame_keyword.split("_")[1])

        with open(input_path,'rb') as f:
            data=pickle.load(f)
        data['normal_joints'] = data['normal_joints'][start_frame:end_frame,...]
        data['lhand_norms'] = data['lhand_norms'][start_frame:end_frame,...]
        data['rhand_norms'] = data['rhand_norms'][start_frame:end_frame,...]
        data['hand_shape_vecs'] = data['hand_shape_vecs'][start_frame:end_frame,...]
        assert data['pose_embeddings'].shape[0]==data['joint_angles'].shape[0]==\
        data['joint_masks'].shape[0]==data['ges_labels'].shape[0]==data['normal_joints'].shape[0]==\
        data['lhand_norms'].shape[0]==data['rhand_norms'].shape[0]==data['hand_shape_vecs'].shape[0]
        with open(output_path,'wb') as f:
            pickle.dump(data, f,protocol=2)