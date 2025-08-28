import subprocess
import torch
import torch.nn.functional as F
import os
import cv2
from utils import JOINT_NAMES
from my_single_frame import fit_single_img

# joint index
L_WRIST = 20
L_INDEX = 25
L_MIDDLE = 28
L_RING = 34
L_PINKY = 31
R_WRIST = 21
R_INDEX = 40
R_MIDDLE = 43
R_RING = 49
R_PINKY = 46

joint2index ={name:i for i,name in enumerate(JOINT_NAMES)}


def compute_gesture_coord(joints):
    # Define face triangle points
    p1 = joints[joint2index["left_eye_brow3"]]
    p2 = joints[joint2index["right_eye_brow3"]]
    p3 = joints[joint2index["lip_bottom"]]

    # Face normal (Z axis): cross product of cheek→cheek and cheek→chin
    v1 = p2 - p1
    v2 = p3 - p1
    z_axis = F.normalize(torch.cross(v1, v2), dim=0)
    # origin
    origin = (joints[joint2index["right_shoulder"]] + joints[joint2index["left_shoulder"]]) / 2
    # X axis: left shoulder → right shoulder
    x_init = F.normalize(joints[joint2index["right_shoulder"]] - joints[joint2index["left_shoulder"]], dim=0)
    # Y axis: orthogonal to Z and initial X
    y_axis = F.normalize(torch.cross(z_axis, x_init), dim=0)
    # Re-orthogonalize X from Y and Z
    x_axis = F.normalize(torch.cross(y_axis, z_axis), dim=0)
    # Final rotation matrix (columns are x, y, z axes)
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # shape (3, 3)
    return origin,R


def classify(normal,refs):
    sims = {k: F.cosine_similarity(normal, v, dim=0).item() for k, v in refs.items()}
    label = max(sims, key=sims.get)
    return label, sims


def compute_palm_normal(joints, wrist_idx, index_base_idx, pinky_base_idx,is_left=True):
    p0 = joints[wrist_idx]
    p1 = joints[index_base_idx]
    p2 = joints[pinky_base_idx]
    v1 = p1 - p0
    v2 = p2 - p0
    if is_left:
        normal = torch.cross(v2, v1)
    else:
        normal = torch.cross(v1, v2)
    normal = torch.nn.functional.normalize(normal, dim=0)
    return normal


def compute_palm_center(joints, indices):
    points = joints[indices]
    center = points.mean(dim=0)
    return center


def get_origin_hand_norm(smplx_joints):
    joints = torch.tensor(smplx_joints)  # tensor, shape: (127, 3)
    with torch.no_grad():
        origin, coord_transform = compute_gesture_coord(joints)

    # Left hand
    left_normal = compute_palm_normal(joints, L_WRIST, L_INDEX, L_PINKY, is_left=True)  # shape [3,]
    left_center = compute_palm_center(joints, [L_WRIST, L_INDEX, L_MIDDLE, L_RING, L_PINKY])-origin
    # Right hand
    right_normal = compute_palm_normal(joints, R_WRIST, R_INDEX, R_PINKY, is_left=False)
    right_center = compute_palm_center(joints, [R_WRIST, R_INDEX, R_MIDDLE, R_RING, R_PINKY])-origin

    result=dict(origin=origin.numpy(),R=coord_transform.numpy(),
                lhand_norm=left_normal.numpy(),rhand_norm=right_normal.numpy(),
                lhand_cen=left_center.numpy(),rhand_cen=right_center.numpy())
    """
    # Define references
    refs = {
        'right': torch.tensor([1.0, 0.0, 0.0]),
        'left': torch.tensor([-1.0, 0.0, 0.0]),
        'down': torch.tensor([0.0, 1.0, 0.0]),
        'up': torch.tensor([0.0, -1.0, 0.0]),
        'inside': torch.tensor([0.0, 0.0, -1.0]),
        'outside': torch.tensor([0.0, 0.0, 1.0])
    }
    # judge if the hand if pointing towards self
    self_ref_left = joints[joint2index['spine3']] - left_center  # spine 3 is roughly the chest
    self_ref_right = joints[joint2index['spine3']] - right_center
    self_score_left = F.cosine_similarity(left_normal, self_ref_left, dim=0).item()
    self_score_right = F.cosine_similarity(right_normal, self_ref_right, dim=0).item()
    self_label_left = "self" if self_score_left > 0.7 else ""  # less than 45 degrees
    self_label_right = "self" if self_score_right > 0.7 else ""  # less than 45 degrees

    # transform palm normals to new coordinates
    left_normal_local = torch.matmul(coord_transform.T, left_normal)
    right_normal_local = torch.matmul(coord_transform.T, right_normal)
    # classify palm orientations
    label_left, sims_left = classify(left_normal_local, refs)
    label_right, sims_right = classify(right_normal_local, refs)
    text = f'left palm orientation: {label_left},{self_label_left}, score:{max(sims_left.values()):.2f}, self_score:{self_score_left:.2f}'
    print(text)
    text = f'right palm orientation: {label_right},{self_label_right}, score:{max(sims_right.values()):.2f}, self_score:{self_score_right:.2f}'
    print(text)
    """
    return result



if __name__ == "__main__":
    # Define references
    refs = {
        'right': torch.tensor([1.0, 0.0, 0.0]),
        'left': torch.tensor([-1.0, 0.0, 0.0]),
        'down': torch.tensor([0.0, 1.0, 0.0]),
        'up': torch.tensor([0.0, -1.0, 0.0]),
        'inside': torch.tensor([0.0, 0.0, -1.0]),
        'outside': torch.tensor([0.0, 0.0, 1.0])
    }
    img_folder = "D:/2025_smplx/smplify-x/inputs/images"
    img_names = os.listdir(img_folder)
    cache_folder="D:/2025_smplx/smplify-x/openpose_cache/"

    out_folder ="D:/2025_smplx/smplify-x/output_0602"
    openpose_wd = "D:/2025_openpose/openpose"
    openpose_cmd = ["D:/2025_openpose/openpose/bin/OpenPoseDemo.exe",
                    "--image_dir", f"{img_folder}",
                    "--write_json", f"{cache_folder}",
                    "--face", "--hand"
                    ]
    # Run the command in the specified working directory
    try:
        subprocess.run(openpose_cmd, cwd=openpose_wd, check=True)
    except subprocess.CalledProcessError as e:
        print("OpenPose execution failed:", e)

    for img_name in img_names:
        img_fn = f"{img_folder}/{img_name}"
        image = cv2.imread(img_fn)
        out_name = f"{out_folder}/{img_name}"
        smplx_model,camera = fit_single_img(img_name,img_folder,cache_folder)
        # Get the 127 joints
        joints = smplx_model.joints[0]  # tensor, shape: (127, 3)
        with torch.no_grad():
            origin,coord_transform = compute_gesture_coord(joints)
            to_right = origin+torch.matmul(coord_transform, refs['right']*0.1)
            to_down = origin+torch.matmul(coord_transform, refs['down']*0.1)
            to_outside = origin+torch.matmul(coord_transform, refs['outside']*0.1)
        # Left hand
        left_normal = compute_palm_normal(joints, L_WRIST, L_INDEX, L_PINKY,is_left=True) # shape [3,]
        left_center = compute_palm_center(joints, [L_WRIST, L_INDEX,L_MIDDLE,L_RING,L_PINKY])
        # Right hand
        right_normal = compute_palm_normal(joints, R_WRIST, R_INDEX, R_PINKY,is_left=False)
        right_center = compute_palm_center(joints, [R_WRIST, R_INDEX, R_MIDDLE,R_RING,R_PINKY])
        points = torch.stack([left_center,left_center+left_normal*0.1,
                              right_center,right_center+right_normal*0.1,
                              origin,to_right,
                              origin,to_down,
                              origin,to_outside,
                              joints[joint2index['spine3']]],
                              dim=0).unsqueeze(0)
        with torch.no_grad():
            camera = camera.to(points.device)
            projected_points = camera(points).to(torch.int64).squeeze(0).detach().numpy()

        # judge if the hand if pointing towards self
        self_ref_left = joints[joint2index['spine3']] - left_center # spine 3 is roughly the chest
        self_ref_right = joints[joint2index['spine3']] - right_center
        self_score_left = F.cosine_similarity(left_normal, self_ref_left, dim=0).item()
        self_score_right = F.cosine_similarity(right_normal, self_ref_right, dim=0).item()
        self_label_left = "self" if self_score_left>0.7 else "" # less than 45 degrees
        self_label_right = "self" if self_score_right > 0.7 else ""  # less than 45 degrees

        # transform palm normals to new coordinates
        left_normal_local = torch.matmul(coord_transform.T, left_normal)
        right_normal_local = torch.matmul(coord_transform.T, right_normal)
        # classify palm orientations
        label_left, sims_left = classify(left_normal_local,refs)
        label_right, sims_right = classify(right_normal_local,refs)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        y_offset = 20
        text = f'left palm orientation: {label_left},{self_label_left}, score:{max(sims_left.values()):.2f}, self_score:{self_score_left:.2f}'
        cv2.putText(image, text, (10, y_offset), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        text = f'right palm orientation: {label_right},{self_label_right}, score:{max(sims_right.values()):.2f}, self_score:{self_score_right:.2f}'
        cv2.putText(image, text, (10, y_offset+20), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)

        # draw left normal
        cv2.arrowedLine(image,tuple(projected_points[0]),tuple(projected_points[1]),(0, 255, 0), 2)
        # draw right normal
        cv2.arrowedLine(image, tuple(projected_points[2]), tuple(projected_points[3]), (0, 255, 0), 2)
        # draw new coordinates
        coord_color = [(0, 255, 127),
                        (255, 0, 127),
                       (0, 0, 255),
                       ]
        for i in range(3):
            axis_start,axis_end = projected_points[4+2*i], projected_points[5+2*i]
            cv2.arrowedLine(image, tuple(axis_start), tuple(axis_end), coord_color[i], 2)
        cv2.circle(image,tuple(projected_points[10]),5,(0, 0, 255),5)

        cv2.imwrite(out_name,image)
        # tensor shape [1,4,2]
        print("success")

