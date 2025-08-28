import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hand_cropper import get_vit_vec,HandEncoder
from ultralytics import YOLO

if __name__ == "__main__":
    vit_model = HandEncoder(vit_pth_path='E:/data/gestureDataset/VitB16.pth').to('cuda')
    yolo_model = YOLO("E:/data/gestureDataset/YOLOv10x_hands.pt").to('cuda')
    train_dir="E:/data/smplx_multisimo/ready_for_annotation"
    save_dir = "E:/data/smplx_multisimo/vit_pca.pkl"
    vit_vec_list=[]
    vec_num=0
    for img_fn in os.listdir(train_dir):
        img_keyword=os.path.splitext(img_fn)[0]
        fn_info = img_keyword.split('_')
        n_frame=int(fn_info[4][1:])
        if n_frame == 2 or n_frame==3:
            img_path=os.path.join(train_dir,img_fn)
            vit_vec=np.squeeze(get_vit_vec(img_path,yolo_model=yolo_model,vit_model=vit_model)) #[768,]
            if len(vit_vec.shape) >0:
                print(vit_vec.shape)
                vit_vec_list.append(vit_vec)
                vec_num+=1
    print(f"\nVit vec extraction complete.\n{vec_num} vectors are added!")

    # train PCA
    features = np.stack(vit_vec_list,axis=0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=128)
    pca.fit(features_scaled)
    print(pca.components_)
    with open(save_dir,'wb') as f:
        pickle.dump(pca,f)
    print("PCA complete.")


