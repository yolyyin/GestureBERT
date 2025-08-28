import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from BERTdataset import BERTDataset
from BERTmodel import BERT
from GesGroupTrainer import GesGroupTrainer, save_loss_plot


if __name__ == "__main__":
    bert_pretrain=None#"/home/ubuntu/Documents/2025_smplx/smplify-x/output_0807_4/bert_trained.model.ep5000.pth"
    ges_pretrain="/home/ubuntu/Documents/2025_smplx/smplify-x/output_0808_ges_1/gesGroup.model.ep500.pth"
    train_fn_dir = "/home/ubuntu/Documents/data/smplx_multisimo/train_0808"
    test_fn_dir="/home/ubuntu/Documents/data/smplx_multisimo/val_0808"
    eval_dir = "/home/ubuntu/Documents/data/smplx_multisimo/train_0808"

    output_dir= "/home/ubuntu/Documents/2025_smplx/smplify-x/output_0808_ges_eval_train"
    os.makedirs(output_dir,exist_ok=True)
    output_path= "/home/ubuntu/Documents/2025_smplx/smplify-x/output_0808_ges_1/gesGroup.model"
    plt_save_path = "/home/ubuntu/Documents/2025_smplx/smplify-x/output_0808_ges_1/loss.png"
    seq_len=32 # should be multiples of n_frame_per_file
    n_frame_per_file=16
    batch_size=128
    num_workers=1

    #n_joints=22
    pose_embed_size=454
    pose_mask_size = 62
    hidden=256
    n_layers = 8
    attn_heads=8
    #bert_alpha=1.0
    #bert_beta=0.2

    epochs=1010
    log_freq=10
    save_freq=50
    lr=1e-3
    warmup_steps=10000
    init_lr_scale=0.05
    adam_weight_decay=0.01
    adam_beta1=0.9
    adam_beta2=0.99

    print(f"Loading Train Dataset: {train_fn_dir}")
    train_dataset = BERTDataset(train_fn_dir, seq_len=seq_len,n_frame_file=n_frame_per_file,full_seq=False)

    print(f"Loading Test Dataset: {test_fn_dir}")
    test_dataset = BERTDataset(test_fn_dir, seq_len=seq_len,n_frame_file=n_frame_per_file,full_seq=False) \
        if test_fn_dir is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,dropout=0.3)
    if bert_pretrain is not None:
        state_dict = torch.load(bert_pretrain)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("bert"):
                new_key=key[5:]
                #print(new_key)
                new_state_dict[new_key] = value
        bert.load_state_dict(new_state_dict)
    #print("Building BERT model")
    #if bert_pretrain is not None:
    #    bert = torch.load(bert_pretrain, weights_only=False)
    #else:
    #    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,dropout=0.1)

    print("Creating BERT Trainer")
    trainer = GesGroupTrainer(bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          warmup_steps=warmup_steps,init_lr_scale=init_lr_scale,num_epochs=epochs,
                          with_cuda=True, log_freq=log_freq,pretrain_model=ges_pretrain)

    """
    print("Training Start")
    train_loss_list=[]
    test_loss_list=[]
    train_acc_list=[]
    test_acc_list=[]
    for epoch in range(epochs):
        trainer.train(epoch,train_loss_list,train_acc_list)
        if test_data_loader is not None:
            trainer.test(epoch, test_loss_list,test_acc_list)

        if epoch % save_freq == 0:
            val_plt_path = os.path.join(output_dir,f"loss_epoch_{epoch}.png")
            save_loss_plot(val_plt_path,train_loss_list,test_loss_list,train_acc_list,test_acc_list)
            trainer.save(epoch, output_path)

    save_loss_plot(plt_save_path, train_loss_list, test_loss_list,train_acc_list,test_acc_list)
    trainer.save(epochs, output_path)
    per500_train_loss = train_loss_list[0:epochs:100]
    per500_train_acc=train_acc_list[0:epochs:100]
    per500_test_loss = test_loss_list[0:epochs:100]
    per500_test_acc=test_acc_list[0:epochs:100]
    table = np.array([per500_train_loss, per500_test_loss,per500_train_acc,per500_test_acc])
    title = "McNeil gesture type prediction"
    c_titles = [f"epoch_{i}" for i in range(0, epochs, 100)]
    r_titles = ["train loss(cross entropy)", "test loss(cross entropy)","train accuracy","test accuracy"]
    df = pd.DataFrame(table, columns=c_titles, index=r_titles)
    with open(f'{output_dir}/final_stats_ges.csv', 'w') as f:
        f.write(title + '\n')  # Write title
        df.to_csv(f)
    print("Final training statistics saved")
    """
    print("Final evaluation...")
    print(f"Loading eval Dataset: {eval_dir}")
    eval_dataset = BERTDataset(eval_dir, seq_len=seq_len, n_frame_file=n_frame_per_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    trainer.evaluate_and_plot(eval_dataloader, save_dir=output_dir)
    print("Final evaluation done")
    print("-----GesGroupModel Successfully trained---------------")