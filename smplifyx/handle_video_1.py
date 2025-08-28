import time
import os
import pympi
from moviepy import VideoFileClip

import cv2
import pickle
import logging
from my_single_frame import prepare_model,get_joints
import subprocess
import numpy as np
from my_fit_frame import fit_single_frame
from data_parser import read_keypoints
from ultralytics import YOLO
from palm_orientation import get_origin_hand_norm
from hand_cropper import HandEncoder,get_vit_vec

TYPE_ICONIC=1
TYPE_SYMBOLIC=2
TYPE_DEICTIC=3
TYPE_BEAT=4
TYPE_OTHER=0


def extract_frames(video_path, output_dir, target_fps=None):
    # extract video name key word
    video_name = os.path.basename(video_path)
    video_keyword = os.path.splitext(video_name)[0]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get original video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    if target_fps is not None:
        frame_interval = int(round(original_fps / target_fps))
    else:
        frame_interval=1

    frame_count = 0
    saved_count = 0
    frame_fn_sequence=[]

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save frame if it's on the target interval
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            frame_fn_sequence.append(filename)
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to {output_dir}.")
    return frame_fn_sequence,video_keyword


def clip_video_by_elan(video_path, eaf_path, tier_name, output_dir, buffer_seconds=30):
    # Load the ELAN file
    eaf = pympi.Elan.Eaf(eaf_path)

    if tier_name not in eaf.tiers:
        raise ValueError(f"Tier '{tier_name}' not found in the ELAN file.")

    video_keyword = os.path.splitext(os.path.basename(video_path))[0]
    # Load the video file
    video = VideoFileClip(video_path)
    video_duration = video.duration  # in seconds

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Get annotations in the tier
    annotations = eaf.get_annotation_data_for_tier(tier_name)

    for idx, (start_ms, end_ms, _) in enumerate(annotations):
        # Convert ms to seconds
        start_time = max((start_ms / 1000) - buffer_seconds, 0)
        end_time = min((end_ms / 1000) + buffer_seconds, video_duration)

        # Extract subclip
        subclip = video.subclipped(start_time, end_time)

        # Save the clip
        output_filename = os.path.join(output_dir, f"{video_keyword}_clip_{idx:04d}.mp4")
        subclip.write_videofile(output_filename, codec='libx264', audio_codec='aac')

    print(f"Done clipping {video_path}! {len(annotations)} clips saved to '{output_dir}'.")


if __name__=="__main__":
    logging.basicConfig(
        filename="/home/ubuntu/Documents/2025_smplx/smplify-x/output_0728/my_log.log",
        level= logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    )
    vit_model = HandEncoder(vit_pth_path='/home/ubuntu/Documents/2025_smplx/smplify-x/models/VitB16.pth').to('cuda')
    yolo_model = YOLO("/home/ubuntu/Documents/2025_smplx/smplify-x/models/YOLOv10x_hands.pt").to('cuda')
    with open('/home/ubuntu/Documents/data/smplx_multisimo/vit_pca.pkl', 'rb') as f:
        pca_model = pickle.load(f)

    n_frame_per_file=16
    FPS=30
    buffer_seconds=0
    origin_video_dir="/home/ubuntu/Documents/data/smplx_multisimo/original_videos_1"
    origin_video_names = [os.path.splitext(fn)[0] for fn in os.listdir(origin_video_dir)]
    elan_anno_dir="/home/ubuntu/Documents/data/smplx_multisimo/eafs"
    video_dir = "/home/ubuntu/Documents/data/smplx_multisimo/new_videos_1"
    cache_dir = '/home/ubuntu/Documents/data/smplx_multisimo/new_caches_1'
    out_folder = '/home/ubuntu/Documents/data/smplx_multisimo/new_smplx_poses'
    #os.makedirs(video_dir,exist_ok=True)
    #os.makedirs(cache_dir,exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)


    """
    # clip videos
    start=time.time()
    for fn in os.listdir(origin_video_dir):
        v_path = os.path.join(origin_video_dir,fn)
        id_str=os.path.splitext(fn)[0]
        ids = id_str.split('_')
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
            clip_video_by_elan(v_path, eaf_path, tier_name, video_dir, buffer_seconds=0)
        except Exception as e:
            print(f"Clipping {v_path} failed:", e)
            continue
    end = time.time()
    print(f"Video preprocess completes. Time spent:{end - start:.4f} seconds")
    """

    clip2ges = {}
    start = time.time()
    for video_keyword in sorted(origin_video_names):
        ids = video_keyword.split('_')
        person_id, session_id = "", ""
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

            for idx, (start_ms, end_ms, anno_str) in enumerate(annotations):
                clip_keyword = f"{video_keyword}_clip_{idx:04d}"
                if anno_str=="Iconic":
                    anno_value=TYPE_ICONIC
                elif anno_str=="Symbolic":
                    anno_value=TYPE_SYMBOLIC
                elif anno_str=="Deictic":
                    anno_value=TYPE_DEICTIC
                elif anno_str=="Beat":
                    anno_value=TYPE_BEAT
                else:
                    anno_value=TYPE_OTHER

                duration_ms = (end_ms - start_ms + 2 * buffer_seconds * 1000.0)
                num_frames = int(np.ceil(duration_ms * FPS / 1000.0))
                gesture_array = np.zeros(num_frames, dtype=np.int64)

                start_frame = int(np.floor(buffer_seconds * FPS))
                end_frame = int(np.ceil(((end_ms - start_ms) / 1000.0 + buffer_seconds) * FPS))
                gesture_array[start_frame:end_frame] = 1*anno_value

                # save gesture_array of every clip
                clip2ges[clip_keyword] = gesture_array
        except  Exception as e:
            print(f"Can't extract ges group info from file {eaf_path} with tier {tier_name}", e)
            exit(1)


    for video_name in sorted(os.listdir(video_dir)):

        video_path = os.path.join(video_dir,video_name)
        folder_name = os.path.splitext(video_name)[0]
        img_folder = os.path.join(cache_dir,folder_name,'images')
        kp_folder = os.path.join(cache_dir,folder_name,'keypoints')
        if not os.path.isdir(kp_folder):
            print(f"\nSkipping {video_name}...No keypoints cache is found....")
            continue

        os.makedirs(img_folder,exist_ok=True)
        os.makedirs(kp_folder,exist_ok=True)

        img_paths,video_keyword = extract_frames(video_path, img_folder, target_fps=FPS)
        """
        openpose_wd = "/home/ubuntu/openpose"
        openpose_cmd = ["./build/examples/openpose/openpose.bin",
                        "--image_dir", f"{img_folder}",
                        "--write_json", f"{kp_folder}",
                        #"--write_images", f"{kp_folder}",
                        "--face", "--hand"
                        ]
        # Run the command in the specified working directory
        try:
            print("OpenPose detection starts ...")
            start = time.time()
            subprocess.run(openpose_cmd, cwd=openpose_wd, check=True)
            end = time.time()
            print(f"OpemPose execution succeeds. Time spent:{end-start:.4f} seconds")
        except subprocess.CalledProcessError as e:
            print("OpenPose execution failed:", e)
        #time.sleep(30)
        """

        save_dict={}
        pose_embedding_list=[]
        pose_joint_angle_list=[]
        joint_mask_list=[]
        normal_joint_list=[]
        lnorm_list = []
        rnorm_list = []
        vit_vec_list = []
        start_frame=1
        first_frame = True
        prev_pose_embedding = None
        prev_camera_trans = None
        prev_body_global_orient = None
        prev_betas=None
        for frame_idx,img_path in enumerate(img_paths):
            start = time.time()
            print(f"\nSMPLX inference for Frame {frame_idx+1} starts...")
            (neutral_model, camera, joint_weights, dtype,
             shape_prior, expr_prior, body_pose_prior,
             left_hand_prior, right_hand_prior, jaw_prior, angle_prior, args) = prepare_model()
            args.pop('save_meshes')

            img_name = os.path.basename(img_path)
            keypoints_name=f"{os.path.splitext(img_name)[0]}_keypoints.json"
            keypoints_fn = os.path.join(kp_folder,keypoints_name)

            img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0

            # read openpose keypoints
            keyp_tuple = read_keypoints(keypoints_fn)
            if len(keyp_tuple.keypoints) < 1:
                print(f"WARNING for {img_path}: OpenPose keypoint is not found!")
                continue
            keypoints = np.stack(keyp_tuple.keypoints)

            #out_mesh_fn = f"{out_folder}/{os.path.splitext(img_name)[0]}.obj"

            # fit the smplx model, body_pose shape [1,21,3]
            result, body_pose, visible_joints_mask, loss = fit_single_frame(img, keypoints[0:1, :, :],
                                                 body_model=neutral_model,
                                                 camera=camera,
                                                 joint_weights=joint_weights,
                                                 dtype=dtype,
                                                 shape_prior=shape_prior,
                                                 expr_prior=expr_prior,
                                                 body_pose_prior=body_pose_prior,
                                                 left_hand_prior=left_hand_prior,
                                                 right_hand_prior=right_hand_prior,
                                                 jaw_prior=jaw_prior,
                                                 angle_prior=angle_prior,
                                                 save_meshes=False,
                                                 first_frame=first_frame, # by yty
                                                 prev_pose_embedding=prev_pose_embedding, # by yty
                                                 prev_camera_trans=prev_camera_trans,
                                                 prev_body_global_orient=prev_body_global_orient,
                                                 prev_betas=prev_betas,
                                                 **args)


            if first_frame:
                first_frame = False

            body_pose_embedding = result['body_pose'] # shape [1,32]
            prev_pose_embedding = body_pose_embedding
            prev_camera_trans = result['camera_translation']
            prev_body_global_orient = result['global_orient']
            prev_betas=result['betas']
            logging.info("********")
            logging.info(f"img_name:{img_name}")
            logging.info(f"camera_trans:{prev_camera_trans}")
            logging.info(f"body_orient:{prev_body_global_orient}")
            logging.info(f"body_betas:{prev_betas}")
            logging.info(f"body_final_loss_val:{loss}")

            joints = get_joints(result,body_pose)[0] #[127,3]
            print(f"\nJoints shape:{joints.shape}")
            print(f"\nMask shape:{visible_joints_mask.shape}")

            # get palm normal vector, normal joints and vit vectors
            hand_ori_result = get_origin_hand_norm(joints)
            lhand_n = np.concatenate((hand_ori_result["lhand_norm"], hand_ori_result["lhand_cen"]))  # [6,]
            rhand_n = np.concatenate((hand_ori_result['rhand_norm'], hand_ori_result['rhand_cen']))  # [6,]
            origin = hand_ori_result['origin']  # [3,]
            normal_joints = joints - origin[None, :]  # [127,3]
            vit_vec_l = get_vit_vec(img_path, yolo_model, vit_model, 'cuda')  # [768]
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

            pose_joint_angle_list.append(joints)
            pose_embedding_list.append(body_pose_embedding)
            joint_mask_list.append(visible_joints_mask)
            normal_joint_list.append(normal_joints)
            lnorm_list.append(lhand_n)
            rnorm_list.append(rhand_n)
            vit_vec_list.append(vit_vec)

            # save if list length == n_frame_per_file
            if len(pose_embedding_list) == n_frame_per_file or frame_idx >= len(img_paths)-1:
                pose_embeddings = np.concatenate(pose_embedding_list, axis=0)  # shape [T,32]
                pose_joint_angles = np.stack(pose_joint_angle_list, axis=0)  # [T,127,3]?
                #print(pose_joint_angles.shape)
                joint_masks = np.concatenate(joint_mask_list, axis=0)# [T,118]
                norm_joints = np.stack(normal_joint_list, axis=0)  # [T,127,3]
                #print(norm_joints.shape)
                lnorms = np.stack(lnorm_list, axis=0)  # [T,6]
                rnorms = np.stack(rnorm_list, axis=0)  # [T,6]
                vit_vecs = np.stack(vit_vec_list, axis=0)  # [T,256]
                ges_array = clip2ges[video_keyword][start_frame-1:frame_idx+1]  # shape [T]
                save_dict['pose_embeddings'] = pose_embeddings
                save_dict['joint_angles'] = pose_joint_angles
                save_dict['joint_masks'] = joint_masks
                save_dict['ges_labels'] = ges_array
                save_dict['normal_joints'] = norm_joints
                save_dict['lhand_norms'] = lnorms
                save_dict['rhand_norms'] = rnorms
                save_dict['hand_shape_vecs'] = vit_vecs
                save_name = f"{video_keyword}.{start_frame:06d}_{frame_idx+1:06d}.pkl"
                save_fn = os.path.join(out_folder,save_name)

                with open(save_fn,'wb') as f:
                    pickle.dump(save_dict,f,protocol=2)

                # clear dict and list
                start_frame = frame_idx + 2
                pose_embedding_list.clear()
                pose_joint_angle_list.clear()
                joint_mask_list.clear()
                normal_joint_list.clear()
                lnorm_list.clear()
                rnorm_list.clear()
                vit_vec_list.clear()
                save_dict.clear()

            end = time.time()
            print(f"\nFrame {frame_idx+1} inference succeeds. Time spent:{end - start:.4f} seconds")

    print("All Done!")








