import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from BERTdataset import BERTDataset
from BERTmodel import GesGroupModel, BERT
from BERToptim import ScheduledOptim
from BERToptim import CosineAnnealingOptim

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import pandas as pd

import tqdm
import matplotlib
matplotlib.use('Agg')  # non-GUI backend (no windows)
import matplotlib.pyplot as plt


def save_loss_plot(save_path, train_loss, val_loss, train_acc,val_acc):
    figure1, ax = plt.subplots()
    ax.plot(train_loss, color='tab:blue', label="training loss")
    ax.plot(val_loss, color='tab:red', label="validation loss")
    ax.plot(train_acc, color='tab:green', label="training accuracy")
    ax.plot(val_acc, color='tab:pink', label="validation accuracy")
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    ax.legend()
    figure1.savefig(save_path)
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')


class GesGroupTrainer:
    """
    BERTTrainer make the pretrained BERT model

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2.?

    """

    def __init__(self, bert: BERT,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
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

        self.bert = bert
        self.model = GesGroupModel(bert)
        if pretrain_model is not None:
            self.model.load_state_dict(torch.load(pretrain_model))
        self.model.to(self.device)

        for name, param in self.model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the AdamW optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, init_lr_scale=init_lr_scale, n_warmup_steps=warmup_steps)
        # cosineAnnealing optimizer
        #total_steps = len(train_dataloader) * num_epochs if train_dataloader is not None else 10000
        #self.optim_schedule = CosineAnnealingOptim(self.optim, total_steps)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch,loss_list,acc_list=None):
        self.model.train()
        self.iteration(epoch, self.train_data,loss_list,acc_list)

    def test(self, epoch,loss_list,acc_list):
        self.model.eval()
        self.iteration(epoch, self.test_data,loss_list, acc_list,train=False)
        self.model.train()

    def iteration(self, epoch, data_loader, loss_list,acc_list=None,train=True):
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
        total_correct = 0.0
        total_element = 0.0


        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)

            # 1. get loss and logits from the ges model
            # logits shape [b,5], label shape [b,]
            loss,logits = self.model.forward(data["ges_input"],data["ges_label"])
            # the 0th position of ges_label is padded as 0, use 1th position
            labels=data["ges_label"][:,1]

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # gesture grouping prediction accuracy
            preds = logits.softmax(dim=-1).argmax(dim=-1)
            correct = preds.eq(labels).sum().item()
            avg_loss += loss.item()
            #loss_list.append(loss.item())
            total_correct += correct
            total_element += labels.nelement()

            #if acc_list is not None:
            #    avg_acc = total_correct / total_element
            #    acc_list.append(avg_acc)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        # only update epoch loss and accuracy
        loss_list.append(avg_loss / len(data_iter))
        if acc_list is not None:
            avg_acc = total_correct / total_element
            acc_list.append(avg_acc)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter),"total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path):
        """
        Saving the current fine-tuned bert model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch + ".pth"
        #torch.save(self.model.cpu(), output_path)
        # self.model.to(self.device)
        torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def evaluate_and_plot(self, eval_dataloader, save_dir="./eval_results"):
        class_names = ['other','iconic', 'symbolic', 'deictic','beat']

        #self.model.eval()
        self.model.train()
        all_preds = []
        all_labels = []
        all_probs = []

        #with torch.no_grad():
        for data in eval_dataloader:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)

            _, logits = self.model.forward(data["ges_input"],data["ges_label"])
            probs = torch.softmax(logits, dim=-1).detach()
            #probs = logits.detach()
            preds = probs.argmax(dim=-1)
            labels = data["ges_label"][:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close()

        print("Classification Report:")
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        print(pd.DataFrame(report_dict).transpose())

        # Save to CSV
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{save_dir}/classification_report.csv")

        # ROC Curve (One-vs-Rest)
        y_true_bin = label_binarize(all_labels, classes=np.arange(len(class_names)))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend(loc="lower right")
        plt.savefig(f"{save_dir}/roc_curve.png")
        plt.close()

        print("Evaluation plots saved to:", save_dir)

if __name__ == "__main__":
    bert_pretrain=None
    ges_pretrain_model="/home/ubuntu/Documents/2025_smplx/smplify-x/output_0729_ges/gesGroup.model.ep4000.pth"
    train_fn_dir =None
    test_fn_dir=None
    eval_dir="/home/ubuntu/Documents/data/smplx_multisimo/new_smplx_poses"

    output_dir="/home/ubuntu/Documents/2025_smplx/smplify-x/output_0729_ges"
    os.makedirs(output_dir,exist_ok=True)
    output_path = "/home/ubuntu/Documents/2025_smplx/smplify-x/output_0729_ges/gesGroup.model"

    seq_len=32 # should be multiples of n_frame_per_file
    n_frame_per_file=16
    batch_size=256
    num_workers=1

    #n_joints=22
    pose_embed_size=454
    pose_mask_size = 62
    hidden=256
    n_layers = 8
    attn_heads=8
    #bert_alpha=1.0
    #bert_beta=0.2

    epochs=4000
    log_freq=10
    save_freq=100
    lr=1e-3
    warmup_steps=2000
    init_lr_scale=0.5
    adam_weight_decay=0.01
    adam_beta1=0.9
    adam_beta2=0.99

    print(f"Loading eval Dataset: {eval_dir}")
    eval_dataset = BERTDataset(eval_dir, seq_len=seq_len, n_frame_file=n_frame_per_file)


    print("Creating Dataloader")
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)



    print("Building BERT model")
    bert = BERT(pose_embed_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads)
    if bert_pretrain is not None:
        bert.load_state_dict(torch.load(bert_pretrain))


    print("Creating BERT Trainer")
    trainer = GesGroupTrainer(bert, train_dataloader=eval_dataloader, test_dataloader=None,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          warmup_steps=warmup_steps,init_lr_scale=init_lr_scale,num_epochs=epochs,
                          with_cuda=True, log_freq=log_freq,pretrain_model=ges_pretrain_model)

    #train_loss_list = []
    #trainer.train(0, train_loss_list)
    #trainer.save(4000,output_path)

    trainer.evaluate_and_plot(eval_dataloader, save_dir=output_dir)