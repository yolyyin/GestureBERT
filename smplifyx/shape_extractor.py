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


def normalize_hand(image, box, wrist_kpt, middle_kpt):
    x1, y1, x2, y2 = box
    hand_crop = image[y1:y2, x1:x2]
    rotated=hand_crop
    pad_h, pad_w = get_padding(rotated, 244, 244, 0)

    #dx = middle_kpt[0] - wrist_kpt[0]
    #dy = middle_kpt[1] - wrist_kpt[1]
    #angle = (np.degrees(np.arctan2(dy, dx)) - 90).astype(float)

    #center = ((x2 - x1) // 2, (y2 - y1) // 2)
    #rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    #rotated = cv2.warpAffine(hand_crop, rot_mat, (hand_crop.shape[1], hand_crop.shape[0]))

    padded = cv2.copyMakeBorder(rotated, 0, max(0, pad_h), 0, max(0, pad_w),
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized = cv2.resize(padded, (224, 224))
    return resized


def load_openpose_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['people'][0]['hand_right_keypoints_2d'], data['people'][0]['hand_left_keypoints_2d']


def process_image(image_path, openpose_json, yolo_model, vit_model, device='cuda'):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.54, 0.499, 0.474], std=[0.234, 0.235, 0.231])
    ])

    # 1. Detect hands with YOLO
    hand_boxes = yolo_model(image_rgb)[0].boxes.xyxy  # format: [(x1, y1, x2, y2), ...]

    # 2. Load OpenPose keypoints
    right_hand_kpts, left_hand_kpts = load_openpose_keypoints(openpose_json)
    kpts = {
        'right': np.array(right_hand_kpts).reshape((-1, 3)),
        'left': np.array(left_hand_kpts).reshape((-1, 3))
    }

    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    output = []

    for i,box in enumerate(hand_boxes):
        x1, y1, x2, y2 = map(int, box)
        #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        """
        hand_side = None
        wrist_idx, middle_idx = 0, 9  # OpenPose hand: 0=wrist, 9=middle fingertip
        for side in ['right', 'left']:
            kpt = kpts[side]
            if kpt[wrist_idx, 2] > 0.2 and kpt[middle_idx, 2] > 0.2:
                kx, ky = kpt[wrist_idx][:2]
                if x1 < kx < x2 and y1 < ky < y2:
                    hand_side = side
                    break

        if not hand_side:
            continue
        """
        # 3. Normalize crop
        #wrist = kpts[hand_side][wrist_idx][:2].astype(int)
        #middle = kpts[hand_side][middle_idx][:2].astype(int)
        wrist=middle=None
        norm_img = normalize_hand(image_bgr, (x1, y1, x2, y2), wrist, middle)

        # 4. Paste small hand image on bottom-right corner
        thumb = cv2.resize(norm_img, (64, 64))
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        thumb_pil = Image.fromarray(thumb_rgb)
        main_w, main_h = pil_image.size
        pad = 0
        pil_image.paste(thumb_pil, (main_w - (64 +pad)*(i+1), main_h -(64 +pad)))

        # 4.Feed into ViT
        input_tensor = transform(Image.fromarray(norm_img)).unsqueeze(0).to(device)
        cls_token, cls_logits = vit_model(input_tensor)
        cls_label = cls_logits.argmax().item()
        cls_vec = cls_token.squeeze().cpu().numpy()

        # 6. Draw on image
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, y1 - 10), f"Class: {cls_label}", fill="white", font=font)

        vec_text = ", ".join([f"{v:.2f}" for v in cls_vec[:5]])
        draw.text((x1, y1 - 25), f"Vec: {vec_text}", fill="white", font=font)

        output.append({
            'bbox': (x1, y1, x2, y2),
            'cls_label': cls_label,
            'shape_vec': cls_vec
        })

    # Save output image
    save_path = image_path.replace('.jpg', '_output.jpg')
    pil_image.save(save_path)
    print(f"Output saved to {save_path}")
    return output


if __name__ == "__main__":
    vit_model = HandEncoder(vit_pth_path='E:/data/gestureDataset/VitB16.pth').to('cuda')
    yolo_model = YOLO("E:/data/gestureDataset/YOLOv10x_hands.pt").to('cuda')
    results = process_image(
        image_path='D:/2025_smplx/smplify-x/output_0704/frame_00053.jpg',
        openpose_json='D:/2025_smplx/smplify-x/output_0704/frame_00053_keypoints.json',
        yolo_model=yolo_model,
        vit_model=vit_model,
        device='cuda'
    )
