import subprocess
import torch
import sys
import cv2
import numpy as np
from my_fit_frame import fit_single_frame
from data_parser import read_keypoints
from cmd_parser import parse_config
from utils import smpl_to_openpose,JointMapper
import smplx
from camera import create_camera
from prior import create_prior


def get_joints(result,body_pose):
    smplx_model = smplx.SMPLX(
        model_path="/home/ubuntu/Documents/2025_smplx/smplify-x/models/smplx",
        gender='neutral',  # can be 'male' or 'female' or 'neutral'
        num_betas=10,
        use_pca=True,
        num_pca_comps=12,  # match SMPLify-X
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_expression=True
    )

    output = smplx_model(
        betas=torch.tensor(result['betas'], dtype=torch.float32),
        body_pose=torch.tensor(body_pose, dtype=torch.float32),
        global_orient=torch.tensor(result['global_orient'], dtype=torch.float32),
        left_hand_pose=torch.tensor(result['left_hand_pose'], dtype=torch.float32),
        right_hand_pose=torch.tensor(result['right_hand_pose'], dtype=torch.float32),
        jaw_pose=torch.tensor(result['jaw_pose'], dtype=torch.float32),
        leye_pose=torch.tensor(result['leye_pose'], dtype=torch.float32),
        reye_pose=torch.tensor(result['reye_pose'], dtype=torch.float32),
        expression=torch.tensor(result['expression'], dtype=torch.float32)
    )
    joints = output.joints
    return joints.detach().cpu().numpy()


def prepare_model():
    args = parse_config(["--config", "/home/ubuntu/Documents/2025_smplx/smplify-x/cfg_files/fit_smplx.yaml",
                         '--visualize="False"',
                         "--model_folder", "/home/ubuntu/Documents/2025_smplx/smplify-x/models",
                         "--vposer_ckpt", "/home/ubuntu/Documents/2025_smplx/smplify-x/vposer/vposer_v2_05",
                         "--part_segm_fn", "/home/ubuntu/Documents/2025_smplx/smplify-x/smplx_parts_segm.pkl"
                         ])
    # check env
    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    joint_mapper = JointMapper(smpl_to_openpose('smplx', use_hands=True,use_face=True,
                                use_face_contour=False,
                                openpose_format="coco25"))
    # create body model
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)
    #print(model_params)
    neutral_model = smplx.create(**model_params)

    # Create the camera object
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        camera = camera.to(device=device)
        neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    optim_weights = np.ones(65 + 2*1 + 51*1 + 17 * 0,
                            dtype=np.float32)
    joint_weights=torch.tensor(optim_weights, dtype=dtype,device=device)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    return (neutral_model,camera,joint_weights,dtype,
            shape_prior,expr_prior,body_pose_prior,
            left_hand_prior,right_hand_prior,jaw_prior,angle_prior,args)


def fit_single_img(img_name,input_folder,cache_folder):
    args = parse_config(["--config", "/home/ubuntu/Documents/2025_smplx/smplify-x/cfg_files/fit_smplx.yaml",
                         '--visualize="False"',
                         "--model_folder", "/home/ubuntu/Documents/2025_smplx/smplify-x/models",
                         "--vposer_ckpt", "/home/ubuntu/Documents/2025_smplx/smplify-x/vposer/vposer_v2_05",
                         "--part_segm_fn", "/home/ubuntu/Documents/2025_smplx/smplify-x/smplx_parts_segm.pkl"
                         ])
    # check env
    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    joint_mapper = JointMapper(smpl_to_openpose('smplx', use_hands=True,use_face=True,
                                use_face_contour=False,
                                openpose_format="coco25"))
    # create body model
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)
    #print(model_params)
    neutral_model = smplx.create(**model_params)

    # Create the camera object
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        camera = camera.to(device=device)
        neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    optim_weights = np.ones(65 + 2*1 + 51*1 + 17 * 0,
                            dtype=np.float32)
    joint_weights=torch.tensor(optim_weights, dtype=dtype,device=device)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    img_fn=f"{input_folder}/{img_name}"
    keypoints_fn=f"{cache_folder}{img_name[:-4]}_keypoints.json"
    img = cv2.imread(f"{img_fn}").astype(np.float32)[:, :, ::-1] / 255.0

    # read openpose keypoints
    keyp_tuple = read_keypoints(keypoints_fn)
    if len(keyp_tuple.keypoints) < 1:
        raise Exception("OpenPose keypoint is not found!")
    keypoints = np.stack(keyp_tuple.keypoints)

    # fit the smplx model
    result,body_pose = fit_single_frame(img, keypoints[0:1,:,:],
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
                             **args)

    # create the smplx model and pass the result params
    smplx_model = smplx.SMPLX(
        model_path="/home/ubuntu/Documents/2025_smplx/smplify-x/models/smplx",
        gender='neutral',  # can be 'male' or 'female' or 'neutral'
        num_betas=10,
        use_pca=True,
        num_pca_comps=12,  # match SMPLify-X
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_expression=True
    )

    smplx_output = smplx_model(
        betas = torch.tensor(result['betas'],dtype=torch.float32),
        body_pose=torch.tensor(body_pose,dtype=torch.float32),
        global_orient = torch.tensor(result['global_orient'],dtype=torch.float32),
        left_hand_pose= torch.tensor(result['left_hand_pose'],dtype=torch.float32),
        right_hand_pose = torch.tensor(result['right_hand_pose'],dtype=torch.float32),
        jaw_pose= torch.tensor(result['jaw_pose'],dtype=torch.float32),
        leye_pose= torch.tensor(result['leye_pose'],dtype=torch.float32),
        reye_pose=torch.tensor(result['reye_pose'],dtype=torch.float32),
        expression=torch.tensor(result['expression'],dtype=torch.float32)
    )
    print('Fitting single img success!')
    return smplx_output,camera


if __name__ == "__main__":
    args = parse_config(["--config", "D:/2025_smplx/smplify-x/cfg_files/fit_smplx.yaml",
                         '--visualize="False"',
                         "--model_folder", "D:/2025_smplx/smplify-x/models",
                         "--vposer_ckpt", "D:/2025_smplx/smplify-x/vposer/vposer_v2_05",
                         "--part_segm_fn", "D:/2025_smplx/smplify-x/smplx_parts_segm.pkl"
                         ])
    # check env
    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    joint_mapper = JointMapper(smpl_to_openpose('smplx', use_hands=True,use_face=True,
                                use_face_contour=False,
                                openpose_format="coco25"))
    # create body model
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)
    #print(model_params)
    neutral_model = smplx.create(**model_params)

    # Create the camera object
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        camera = camera.to(device=device)
        neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    optim_weights = np.ones(65 + 2*1 + 51*1 + 17 * 0,
                            dtype=np.float32)
    joint_weights=torch.tensor(optim_weights, dtype=dtype,device=device)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    img_folder="D:/2025_smplx/smplify-x/inputs/images"
    img_name="0.jpg"
    img_fn=f"{img_folder}/{img_name}"
    out_folder = "D:/2025_smplx/smplify-x/output_0527"
    out_fn=f"{out_folder}/{img_name}"
    cache_folder="D:/2025_smplx/smplify-x/openpose_cache/"
    keypoints_fn=f"{cache_folder}{img_name[:-4]}_keypoints.json"
    img = cv2.imread(f"{img_folder}/{img_name}").astype(np.float32)[:, :, ::-1] / 255.0
    openpose_wd = "D:/2025_openpose/openpose"
    openpose_cmd = ["D:/2025_openpose/openpose/bin/OpenPoseDemo.exe",
                    "--image_dir", f"{img_folder}",
                    "--write_json", f"{cache_folder}",
                    "--face","--hand"
                    ]
    # Run the command in the specified working directory
    try:
        subprocess.run(openpose_cmd, cwd=openpose_wd, check=True)
    except subprocess.CalledProcessError as e:
        print("OpenPose execution failed:", e)

    # read openpose keypoints
    keyp_tuple = read_keypoints(keypoints_fn)
    if len(keyp_tuple.keypoints) < 1:
        raise Exception("OpenPose keypoint is not found!")
    keypoints = np.stack(keyp_tuple.keypoints)

    # fit the smplx model
    result,body_pose = fit_single_frame(img, keypoints[0:1,:,:],
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
                             **args)

    # create the smplx model and pass the result params
    smplx_model = smplx.SMPLX(
        model_path="D:/2025_smplx/smplify-x/models/smplx",
        gender='neutral',  # can be 'male' or 'female' or 'neutral'
        num_betas=10,
        use_pca=True,
        num_pca_comps=12,  # match SMPLify-X
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_expression=True
    )

    output = smplx_model(
        betas = torch.tensor(result['betas'],dtype=torch.float32),
        body_pose=torch.tensor(body_pose,dtype=torch.float32),
        global_orient = torch.tensor(result['global_orient'],dtype=torch.float32),
        left_hand_pose= torch.tensor(result['left_hand_pose'],dtype=torch.float32),
        right_hand_pose = torch.tensor(result['right_hand_pose'],dtype=torch.float32),
        jaw_pose= torch.tensor(result['jaw_pose'],dtype=torch.float32),
        leye_pose= torch.tensor(result['leye_pose'],dtype=torch.float32),
        reye_pose=torch.tensor(result['reye_pose'],dtype=torch.float32),
        expression=torch.tensor(result['expression'],dtype=torch.float32)
    )
    joints = output.joints
    print('success')

"""
    # ---- STEP 1: Convert axis-angle to Euler angles ----
    axis_angle = body_pose[0]
    selected_joint_indices = {
        'Left Shoulder': 16,
        'Right Shoulder': 17,
        'Left Elbow': 18,
        'Right Elbow': 19,
        'Left Wrist': 20,
    }

    # Convert and collect
    joint_angles = { }
    for name, idx in selected_joint_indices.items():
        rot = R.from_rotvec(axis_angle[idx])
        euler = rot.as_euler('zyx', degrees=True)  # Yaw-Pitch-Roll
        joint_angles[name] = euler  # [yaw, pitch, roll]

    # ---- STEP 2: Annotate image ----
    image = cv2.imread(img_fn)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (0, 255, 0)
    thickness = 1

    y_offset = 20
    for i, (name, angles) in enumerate(joint_angles.items()):
        yaw, pitch, roll = angles
        text = f'{name}: YPR [{yaw:.1f}, {pitch:.1f}, {roll:.1f}]'
        y_pos = y_offset + i * 20
        cv2.putText(image, text, (10, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    # ---- STEP 3: Save output ----
    cv2.imwrite(out_fn, image)
    print(f"Saved image with joint YPR annotations to: {out_fn}")
"""
