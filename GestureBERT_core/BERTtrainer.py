import os

import numpy as np
import torch
from torch.optim import AdamW,SGD
from torch.utils.data import DataLoader
import pandas as pd

from BERTdataset import BERTDataset,custom_collate
from BERTmodel import BERTpretrain, BERT
from BERToptim import ScheduledOptim
from BERToptim import CosineAnnealingOptim

import tqdm
import matplotlib
matplotlib.use('Agg')  # non-GUI backend (no windows)
import matplotlib.pyplot as plt

PAD_IDX= -3.0
MASK_IDX= 2.0
UNK_IDX = -2.0
SOS_IDX=3.0


def save_loss_plot(save_path, train_loss, val_loss):
    figure1, ax = plt.subplots()
    ax.plot(train_loss, color='tab:blue', label="training loss")
    ax.plot(val_loss, color='tab:red', label="validation loss")
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    ax.legend()
    figure1.savefig(save_path)
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2.?

    """

    def __init__(self, bert: BERT, pose_embed_size: int,pose_mask_size:int,
                 train_dataloader: DataLoader = None, test_dataloader: DataLoader = None, bert_alpha=1.0,bert_beta=0.4,bert_gamma=0.5,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,init_lr_scale=1.0,num_epochs=1,
                 with_cuda: bool = True, log_freq: int = 10,pretrain_model=None):
        """
        :param bert: BERT model which you want to train
        :param pose_embed_size: pose embedding size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTpretrain(bert, pose_embed_size,pose_mask_size,bert_alpha,bert_beta,bert_gamma)
        if pretrain_model is not None:
            self.model.load_state_dict(torch.load(pretrain_model))
            print("Pretrained model loaded!")
        self.model.to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim = SGD(self.model.parameters(), lr=lr,momentum=0.9)

        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, init_lr_scale=init_lr_scale, n_warmup_steps=warmup_steps)
        # cosineAnnealing optimizer
        #total_steps = len(train_dataloader) * num_epochs if train_dataloader is not None else 10000
        #self.optim_schedule = CosineAnnealingOptim(self.optim, total_steps)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch,loss_list):
        self.model.train()
        self.iteration(epoch, self.train_data,loss_list)

    def test(self, epoch,loss_list):
        self.model.eval()
        self.iteration(epoch, self.test_data,loss_list, train=False)
        self.model.train()

    def get_embed(self, data_loader,av_embed=False):
        """
        loop over the data_loader to get bert embeddings
        return embeddings as a numpy matrix
        """
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Batch extracting BERT embedding...",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        seq_embeds_list=[]
        img_paths_list=[]
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key]=value.to(self.device)
            img_paths=data["img_paths"]
            #print("+++++++++")
            #print(img_paths)
            img_paths_list.append(img_paths)

            # 1. get bert embeddings
            with torch.no_grad():
                bert_embeds = self.model.bert(data["bert_input"]) # [b,t,hidden]
            bert_embeds=bert_embeds.detach().cpu().numpy() # [b,t,hidden]
            if av_embed:
                seq_embeds=np.average(bert_embeds,axis=1) #[b,hidden]
                print()
            else:
                #extract [CLS] seq
                seq_embeds=bert_embeds[:,0,:] #[b,hidden]
            seq_embeds_list.append(seq_embeds)
        seq_embeds=np.concatenate(seq_embeds_list,axis=0) #[n_data,hidden]
        return seq_embeds,img_paths_list

    def iteration(self, epoch, data_loader, loss_list,train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)

            # 1. get loss and bert model
            loss, _,_,_ = self.model.forward(data["bert_input"],data["bert_label"],data["bert_pose_mask"])

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write("\n" + str(post_fix))

        # only update epoch loss
        loss_list.append(avg_loss / len(data_iter))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def save(self, epoch, file_path="output_0612/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch + ".pth"
        # torch.save(self.model.cpu(), output_path)
        # self.model.to(self.device)
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def predict(self, input_seq,seq_len=32,predict_len=16,step=8):
        """
        predict future ges frames using masked token completion strategy
        :param input_seq: tensor of shape [1, len, pose_embed_size]
        :return: tensor of shape [1, predict_len, pose_embed_size]
        """
        self.model.eval()
        input_seq=input_seq.to(self.device)
        pose_embed_size=input_seq.size(-1)

        total_preds=[]
        while len(total_preds) < predict_len//step:
            # Append step-length [MSK] tokens to input
            input_len = input_seq.shape[1]
            print(input_seq[0, input_len-seq_len+step:, 21])
            sos_tokens=torch.full((1,1,pose_embed_size),SOS_IDX,device=self.device)
            msk_tokens=torch.full((1,step,pose_embed_size),MASK_IDX,device=self.device) #[1,step,pose_embed_size]
            model_input=torch.cat([sos_tokens,input_seq[:,input_len-seq_len+step:,:],msk_tokens],dim=1) #[1,seq_len+1,pose_embed_size]
            print(model_input[0,-5:,21])

            # Get prediction
            with torch.no_grad():
                bert_output=self.model.bert(model_input)
                prediction=self.model.pose_predictor(bert_output)

            # Take the last 'step' predictions
            predicted_frames=prediction[:,-step:,:] # [1,step,pose_embed_size]
            total_preds.append(predicted_frames)

            # append predicted_frames to input_seq

            input_seq=torch.cat([input_seq,predicted_frames],dim=1)


        final_preds=torch.cat(total_preds,dim=1) # [1, predict_len, pose_embed_size
        return final_preds,input_seq


if __name__ == "__main__":
    pretrain_model_path = None
    train_fn_dir = "data/multisimo/train"
    test_fn_dir= "data/multisimo/val"

    output_dir="output/pretrain"
    os.makedirs(output_dir,exist_ok=True)
    output_path="output/pretrain/bert_trained.model"
    plt_save_path = "output/pretrain/loss.png"
    seq_len=32 # should be multiples of n_frame_per_file
    n_frame_per_file=16
    batch_size=64
    num_workers=1

    pose_embed_size=454
    pose_mask_size = 62
    hidden=256
    n_layers = 8
    attn_heads=8
    bert_alpha=1.0
    bert_beta=0.05
    bert_gamma=2.0

    epochs=6010
    log_freq=10
    save_freq=100
    lr=1e-6
    warmup_steps=5000
    init_lr_scale=0.3
    adam_weight_decay=0.01
    adam_beta1=0.9
    adam_beta2=0.99

    print(f"Loading Train Dataset: {train_fn_dir}")
    train_dataset = BERTDataset(train_fn_dir, seq_len=seq_len,n_frame_file=n_frame_per_file)

    print(f"Loading Test Dataset: {test_fn_dir}")
    test_dataset = BERTDataset(test_fn_dir, seq_len=seq_len,n_frame_file=n_frame_per_file) \
        if test_fn_dir is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True,collate_fn=custom_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True,collate_fn=custom_collate) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads,dropout=0.1)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, pose_embed_size, pose_mask_size,train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          bert_alpha=bert_alpha, bert_beta=bert_beta,bert_gamma=bert_gamma,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          warmup_steps=warmup_steps,init_lr_scale=init_lr_scale,num_epochs=epochs,
                          with_cuda=True, log_freq=log_freq,pretrain_model=pretrain_model_path)

    print("Training Start")
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(epochs):
        trainer.train(epoch,train_loss_list)
        if test_data_loader is not None:
            trainer.test(epoch, test_loss_list)

        if epoch % save_freq == 0:
            val_plt_path = os.path.join(output_dir,f"loss_epoch_{epoch}.png")
            save_loss_plot(val_plt_path,train_loss_list,test_loss_list)
            trainer.save(epoch, output_path)

    save_loss_plot(plt_save_path, train_loss_list, test_loss_list)
    trainer.save(epochs, output_path)
    per200_train_loss = train_loss_list[0:epochs:500]
    per200_test_loss=test_loss_list[0:epochs:500]
    table=np.array([per200_train_loss,per200_test_loss])
    title="MSE for masked frame prediction"
    c_titles=[f"epoch_{i}" for i in range(0,epochs,500)]
    r_titles=["train loss", "test loss"]
    df = pd.DataFrame(table, columns=c_titles, index=r_titles)
    with open(f'{output_dir}/final_stats.csv', 'w') as f:
        f.write(title + '\n')  # Write title
        df.to_csv(f)
    print("Final training statistics saved")
    print("-----PoseBERT Successfully trained---------------")