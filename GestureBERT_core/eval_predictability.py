import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-GUI backend (no windows)
import matplotlib.pyplot as plt

from BERTdataset import BERTDataset
from BERTmodel import GesGroupModel, BERT
from BERTtrainer import BERTTrainer

PAD_IDX= -3.0
MASK_IDX= 2.0
UNK_IDX = -2.0
SOS_IDX=3.0

L_WRIST=20
R_WRIST=21

if __name__=='__main__':
    pretrain_model_path = "/home/ubuntu/Documents/2025_smplx/smplify-x/output_0806_generate_3/bert_trained.model.ep1200.pth"
    test_fn_dir= "/home/ubuntu/Documents/data/smplx_multisimo/val_0806"

    output_dir="/home/ubuntu/Documents/2025_smplx/smplify-x/output_0806_pred_eval"
    os.makedirs(output_dir,exist_ok=True)
    data_seq_len=64
    n_frame_per_file=16
    batch_size=1
    num_workers=1

    pred_seq_len = 32  # should be multiples of n_frame_per_file
    pred_len=16
    pred_step = 4

    pose_embed_size=454
    pose_mask_size = 62
    hidden=256
    n_layers = 8
    attn_heads=8
    bert_alpha=1.0
    bert_beta=0.0

    epochs=5000
    log_freq=10
    save_freq=100
    lr=1e-5
    warmup_steps=2000
    init_lr_scale=0.5
    adam_weight_decay=0.01
    adam_beta1=0.9
    adam_beta2=0.99

    print(f"Loading Test Dataset: {test_fn_dir}")
    test_dataset = BERTDataset(test_fn_dir, seq_len=data_seq_len,n_frame_file=n_frame_per_file) \
        if test_fn_dir is not None else None

    print("Creating Dataloader")
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=False) \


    print("Building BERT model")
    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, pose_embed_size, pose_mask_size,train_dataloader=None, test_dataloader=test_data_loader,
                          bert_alpha=bert_alpha, bert_beta=bert_beta,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          warmup_steps=warmup_steps,init_lr_scale=init_lr_scale,num_epochs=epochs,
                          with_cuda=True, log_freq=log_freq,pretrain_model=pretrain_model_path)

    baseline_total_error=0.0
    model_total_error=0.0
    cur_clip=""
    for i, data in enumerate(test_data_loader):
        seq_input,seq_label,img_paths = data["bert_input"],data["bert_label"],data['img_paths'] #[1,65,454],[1,65,454]
        # skip the same clip
        clip_name = os.path.basename(img_paths[0][0]).split('.')[0]
        if cur_clip==clip_name:
            continue
        else:
            cur_clip=clip_name
        # reconstruct ground truth
        gt_mask = (seq_label == PAD_IDX).to(torch.long)
        seq_gt= gt_mask*seq_input + (1-gt_mask)*seq_label  # [1,65,454]
        # using the first 28 frame to predict the next 16 frame
        pred_input=seq_gt[:,1:1+pred_seq_len-pred_step,:] # [1,28,454]
        pred_target=seq_gt[:,1+pred_seq_len-pred_step:1+pred_seq_len-pred_step+pred_len,:] # [1,16,454]

        # baseline output, just copy the last frame of pred_input
        baseline_prediction=pred_input[:,-1,:].unsqueeze(1).repeat(1,pred_len,1) # [1,16,454]
        baseline_error=F.mse_loss(baseline_prediction,pred_target,reduction='sum')
        baseline_total_error+= baseline_error.item()/pred_len

        # model output
        model_prediction,_=trainer.predict(pred_input,pred_seq_len,pred_len,pred_step) # [1,16,454]
        model_prediction=model_prediction.to('cpu')
        model_error=F.mse_loss(model_prediction,pred_target,reduction='sum')
        model_total_error+= model_error.item()/pred_len

        # draw the prediction images of left wrist and right wrist
        lhand_gt_seq=seq_gt[0,1:,L_WRIST*3].numpy() #[64,]
        gt_frames=np.arange(1,65)
        lhand_baseline_pred=baseline_prediction[0,:,L_WRIST*3].numpy() #[16,]
        lhand_model_pred=model_prediction[0,:,L_WRIST*3].numpy() #[16,]
        pred_frames=np.arange(1+pred_seq_len-pred_step,1+pred_seq_len-pred_step+pred_len)

        plt.figure(figsize=(8,6))
        plt.plot(gt_frames,lhand_gt_seq,'b-',label='Ground Truth')
        plt.plot(pred_frames,lhand_baseline_pred,'k--',label='Baseline Prediction')
        plt.plot(pred_frames, lhand_model_pred, 'g-', label='Model Prediction')
        plt.xlabel('Frame Number')
        plt.ylabel('Nomalized X Location')
        plt.title('Left Wrist Movement Trajectory on X axis')
        plt.legend()

        plt.savefig(os.path.join(output_dir,clip_name+'.left_wrist.png'))
        plt.close()

        rhand_gt_seq = seq_gt[0, 1:, R_WRIST * 3].numpy()  # [64,]
        rhand_baseline_pred = baseline_prediction[0, :, R_WRIST * 3].numpy()  # [16,]
        rhand_model_pred = model_prediction[0, :, R_WRIST * 3].numpy()  # [16,]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_frames, rhand_gt_seq, 'b-', label='Ground Truth')
        plt.plot(pred_frames, rhand_baseline_pred, 'k--', label='Baseline Prediction')
        plt.plot(pred_frames, rhand_model_pred, 'g-', label='Model Prediction')
        plt.xlabel('Frame Number')
        plt.ylabel('Nomalized X Location')
        plt.title('Right Wrist Movement Trajectory on X axis')
        plt.legend()
        plt.savefig(os.path.join(output_dir, clip_name + '.right_wrist.png'))
        plt.close()

    # calculate average mse error and save to statistic csv
    baseline_mean_error=baseline_total_error/len(test_data_loader)
    model_mean_error=model_total_error/len(test_data_loader)
    statistics=[[baseline_mean_error,model_mean_error]]
    df=pd.DataFrame(statistics,index=['Mean Squared Error of Predicted Frame Embedding'],
                    columns=['Baseline Model', 'GestureBERT'])
    save_fn=os.path.join(output_dir,'prediction_mse.csv')
    df.to_csv(save_fn)
    print("Congrad! prediction evaluation done!")




