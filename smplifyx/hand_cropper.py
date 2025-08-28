import os.path
import pickle
import cv2
import math
import torch
import torchvision.transforms as T
import numpy as np
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import vit_b_16, ViT_B_16_Weights

N_CLS=34


class HandEncoder(torch.nn.Module):
    def __init__(self, vit_pth_path=None):
        super().__init__()
        self.vit = vit_b_16(weights=None,num_classes=N_CLS)  # Don't load default weights

        if vit_pth_path:
            state_dict = torch.load(vit_pth_path, map_location='cpu')['MODEL_STATE']
            vit_state_dict = {}
            for key,value in state_dict.items():
                new_key = '.'.join(key.split('.')[1:])
                vit_state_dict[new_key]=value
            self.vit.load_state_dict(vit_state_dict)


        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            # refer to vit source code: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289
            x = self.vit._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.vit.encoder(x)  # CLS token, shape: (B, 768)
            cls_token = x[:, 0]

            logits = self.vit.heads(cls_token)        # Classification logits
        return cls_token, logits


def get_padding(img, des_h, des_w, margin=5):
    """
    img: ndarray
    des_h, des_w : int
    margin: int
    """
    # get padding size for padding and resize transform, the destination w and h are inputs
    h,w = img.shape[0], img.shape[1]
    scale = min(float(des_h-2*margin)/h, float(des_w-2*margin)/w)
    pad_h, pad_w = math.ceil((des_h-scale*h)/(2*scale)), math.ceil((des_w-scale*w)/(2*scale))
    return pad_h, pad_w


def normalize_hand(image, box):
    x1, y1, x2, y2 = box
    hand_crop = image[y1:y2, x1:x2]
    rotated=hand_crop
    pad_h, pad_w = get_padding(rotated, 244, 244, 0)


    padded = cv2.copyMakeBorder(rotated, 0, max(0, pad_h), 0, max(0, pad_w),
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized = cv2.resize(padded, (224, 224))
    return resized


def load_openpose_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['people'][0]['hand_right_keypoints_2d'], data['people'][0]['hand_left_keypoints_2d']


def process_image(image_path, out_dir,yolo_model, vit_model, device='cuda'):
    img_name=os.path.basename(image_path)
    image_bgr = cv2.imread(image_path)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.54, 0.499, 0.474], std=[0.234, 0.235, 0.231])
    ])

    # 1. Detect hands with YOLO
    hand_boxes = yolo_model(image_path)[0].boxes.xyxy  # format: [(x1, y1, x2, y2), ...]

    for i,box in enumerate(hand_boxes):
        x1, y1, x2, y2 = map(int, box)
        norm_img = normalize_hand(image_bgr, (x1, y1, x2, y2))

        crop = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop)

        # 4.Feed into ViT
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)
        cls_token, cls_logits = vit_model(input_tensor)
        cls_vec = cls_token.squeeze().cpu().numpy()

        # Save croped_hand
        crop_fn = img_name.replace('.jpg', f'_crop_{i}.jpg')
        crop_path = os.path.join(out_dir,crop_fn)
        crop_pil.save(crop_path)
        print(f"Cropped hand saved to {crop_path}")

        # Save cls vec to pickles
        vec_fn= img_name.replace('.jpg', f'_crop_{i}.pkl')
        vec_path=os.path.join(out_dir,vec_fn)
        with open(vec_path, 'wb') as f:
            pickle.dump(cls_vec, f, protocol=2)
            print(f"Vit vector saved to {vec_path}")


def get_vit_vec(image_path, yolo_model, vit_model, device='cuda'):
    image_bgr = cv2.imread(image_path)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.54, 0.499, 0.474], std=[0.234, 0.235, 0.231])
    ])

    # 1. Detect hands with YOLO
    hand_boxes = yolo_model(image_path)[0].boxes.xyxy  # format: [(x1, y1, x2, y2), ...]

    cls_vec_list=[]
    for i,box in enumerate(hand_boxes):
        x1, y1, x2, y2 = map(int, box)
        norm_img = normalize_hand(image_bgr, (x1, y1, x2, y2))

        crop = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop)

        # 4.Feed into ViT
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)
        cls_token, cls_logits = vit_model(input_tensor)
        cls_vec = cls_token.squeeze().cpu().numpy()
        cls_vec_list.append(cls_vec)

    return cls_vec_list



if __name__ == "__main__":
    input_dir="E:/data/smplx_multisimo/vit_vec_data/input"
    output_dir="E:/data/smplx_multisimo/vit_vec_data/output"
    os.makedirs(output_dir,exist_ok=True)
    vit_model = HandEncoder(vit_pth_path='E:/data/gestureDataset/VitB16.pth').to('cuda')
    yolo_model = YOLO("E:/data/gestureDataset/YOLOv10x_hands.pt").to('cuda')
    for img_fn in os.listdir(input_dir):
        img_path=os.path.join(input_dir,img_fn)
        process_image(
            image_path=img_path,
            out_dir=output_dir,
            yolo_model=yolo_model,
            vit_model=vit_model,
            device='cuda'
        )
