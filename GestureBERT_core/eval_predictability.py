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
    pretrain_model_path = "models/bert_trained.model.ep2500.pth"
    test_fn_dir= "data/multisimo/smplx_poses"

    output_dir="output/predictability_eval"
    os.makedirs(output_dir,exist_ok=True)
    data_seq_len=32
    n_frame_per_file=16
    batch_size=1
    num_workers=1

    #pred_seq_len = 32  # should be multiples of n_frame_per_file

    #pred_step = 4

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
        start_frame=os.path.basename(img_paths[0][0]).split('.')[1].split('_')[0]
        end_frame=os.path.basename(img_paths[-1][0]).split('.')[1].split('_')[-1]
        msk_mask= (seq_input == MASK_IDX).to(torch.long) #[1,33,454]
        # reconstruct ground truth
        seq_gt=seq_label  # [1,33,454]

        pred_frames=torch.nonzero((msk_mask)[0,:,0],as_tuple=True)[0]
        pred_frames=pred_frames.tolist() #len=T
        pred_len=len(pred_frames)
        # using the first 28 frame to predict the next 16 frame
        pred_input=seq_input # [1,33,454]
        pred_target=seq_label[:,pred_frames,:] # [1,T,454]

        # baseline output, just copy the previous_frame
        prev_frames=[frame-1 for frame in pred_frames]
        baseline_prediction=pred_input[:,prev_frames,:] # [1,T,454]
        baseline_error=F.mse_loss(baseline_prediction,pred_target,reduction='sum')
        baseline_total_error+= baseline_error.item()/pred_len

        # model output
        with torch.no_grad():
            x=pred_input.to(trainer.device)
            bert_output = trainer.model.bert(x)
            model_prediction = trainer.model.pose_predictor(bert_output) # [1,33,454]
        model_prediction_full=model_prediction.clone().to('cpu') # [1,33,454]
        model_prediction = model_prediction[:,pred_frames,:] # [1,T,454]
        model_prediction=model_prediction.to('cpu')
        model_error=F.mse_loss(model_prediction,pred_target,reduction='sum')
        model_total_error+= model_error.item()/pred_len

        # draw the prediction images of left wrist and right wrist
        lhand_gt_seq=seq_gt[0,1:,L_WRIST*3].numpy() #[32,]
        gt_frames=np.arange(1,33)
        lhand_baseline_pred=baseline_prediction[0,:,L_WRIST*3].numpy() #[T,]
        lhand_model_full=model_prediction_full[0,1:,L_WRIST*3].numpy() #[32,]
        lhand_model_pred=model_prediction[0,:,L_WRIST*3].numpy() #[T,]
        pred_frames=np.array(pred_frames)

        plt.figure(figsize=(8,6))
        plt.plot(gt_frames,lhand_gt_seq,'b-',label='Ground Truth')
        plt.plot(gt_frames,lhand_model_full,'m--',label='Model Prediction (all frames)')
        plt.scatter(pred_frames,lhand_baseline_pred,c='black',marker='x',label='Baseline Prediction (masked frames)')
        plt.scatter(pred_frames, lhand_model_pred, c='green',marker='^', label='Model Prediction (masked frames)')
        plt.xlabel('Frame Number')
        plt.ylabel('Nomalized X Location')
        plt.title('Left Wrist Movement Trajectory on X axis')
        plt.legend()

        plt.savefig(os.path.join(output_dir,f"{clip_name}.{start_frame}_{end_frame}.left_wrist.png"))
        plt.close()

        rhand_gt_seq = seq_gt[0, 1:, R_WRIST * 3].numpy()  # [32,]
        rhand_baseline_pred = baseline_prediction[0, :, R_WRIST * 3].numpy()  # [T,]
        rhand_model_pred = model_prediction[0, :, R_WRIST * 3].numpy()  # [T,]
        rhand_model_full = model_prediction_full[0, 1:, R_WRIST * 3].numpy()  # [32,]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_frames, rhand_gt_seq, 'b-', label='Ground Truth')
        plt.plot(gt_frames, rhand_model_full, 'm--', label='Model Prediction (all frames)')
        plt.scatter(pred_frames, rhand_baseline_pred,c='black',marker='x', label='Baseline Prediction (masked frames)')
        plt.scatter(pred_frames, rhand_model_pred, c='green',marker='^', label='Model Prediction (masked frames)')
        plt.xlabel('Frame Number')
        plt.ylabel('Nomalized X Location')
        plt.title('Right Wrist Movement Trajectory on X axis')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{clip_name}.{start_frame}_{end_frame}.right_wrist.png"))
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




