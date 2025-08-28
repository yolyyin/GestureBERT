import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import pickle
from pympi.Elan import Eaf


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


class PoseBERT(nn.Module):
    def __init__(self, pose_dim, model_dim, num_layers=4, num_heads=8, max_len=512, dropout=0.1):
        super(PoseBERT, self).__init__()
        self.model_dim = model_dim
        self.pose_proj = nn.Linear(pose_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, 1)  # Binary output

    def forward(self, x,labels=None):
        """
        x: [B, T, pose_dim]
        """
        B, T, _ = x.size()
        if T > self.pos_embed.size(1):
            raise ValueError(f"Input sequence too long: {T} > max_len={self.pos_embed.size(1)}")
        x = self.pose_proj(x) + self.pos_embed[:, :T]
        x = self.encoder(x)
        logits = self.classifier(x).squeeze(-1)  # [B, T]
        probs = torch.sigmoid(logits)

        if labels is not None:
            loss = F.binary_cross_entropy(probs, labels.float())
            return logits, loss
        else:
            return logits, probs


def sliding_window_batch(video_tensor,label_tensor,window_size=128, stride=64):
    """
    video_tensor: [T, D] — pose sequence of one video
    label_tensor: [T, 2] — label sequence of one video
    returns: [N, window_size, D] — batch of overlapping chunks
    """
    T, D = video_tensor.size()
    pose_windows,label_windows = [],[]
    for start in range(0, T, stride):
        end = start + window_size
        if end <= T:
            pose_window = video_tensor[start:end]
            label_window = label_tensor[start:end]
        else:
            pad = torch.zeros((end - T, D), device=video_tensor.device)
            pose_window = torch.cat([video_tensor[start:], pad], dim=0)
            label_window = torch.cat([label_tensor[start:], pad], dim=0)
        pose_windows.append(pose_window)
        label_windows.append(label_window)
    return torch.stack(pose_windows),torch.stack(label_windows)  # Shape: [N, window_size, D]


def extract_gesture_labels(eaf_path, tier_name, fps):
    eaf = Eaf(eaf_path)
    annotations = eaf.get_annotation_data_for_tier(tier_name)

    # Determine the full duration of the video using the time slots
    time_slots = eaf.timeslots
    max_time_ms = max(time_slots.values())

    num_frames = int(np.ceil(max_time_ms * fps / 1000.0))
    gesture_array = np.zeros(num_frames, dtype=np.int64)

    for start, end, _ in annotations:
        start_frame = int(np.floor(start * fps / 1000.0))
        end_frame = int(np.ceil(end * fps / 1000.0))
        gesture_array[start_frame:end_frame] = 1

    return gesture_array


# data loading
def get_batch(split,train_data,val_data,train_targets,val_targets):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    targets = train_targets if split == 'train' else val_targets
    pose_windows,label_windows = sliding_window_batch(data,targets,window_size=window_size,stride=stride)
    num_window = pose_windows.shape[0]
    ix = torch.randint(0,num_window, (batch_size,))
    x = torch.stack([pose_windows[i] for i in ix],dim=0)
    y = torch.stack([label_windows[i] for i in ix],dim=0)
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model,train_data,val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,train_data,val_data,train_targets,val_targets)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

max_iters = 1000 #5000
max_epochs= 50
eval_interval = 100
eval_iters = 10
batch_size = 8
window_size=64
stride=32
pose_dim = 32  # from Simplx
model_dim = 512
seq_len = 300   # example length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-5
OUT_DIR = "D:/2025_smplx/smplify-x/output_0603/train_test"
last_save_path = f"{OUT_DIR}/last_v2.pth"
plt_save_path = f"{OUT_DIR}/loss.png"
video_fn = "D:/2025_smplx/smplify-x/inputs/video_test/S02_M001.mp4"
pose_embedding_fn="D:/2025_smplx/smplify-x/output_0603/S02_M001/pose_embeddings.pkl"
target_fn = "D:/2025_smplx/smplify-x/inputs/video_test/S02.eaf"
label_tier_name = "Gestures_M001_S02"
FPS = 10


if __name__ == "__main__":
    # plt style
    plt.style.use('ggplot')
    plt_path = "./"

    torch.manual_seed(1337)
    with open(pose_embedding_fn, "rb") as f:
        pose_data = pickle.load(f) # shape [T,32]
    with open(target_fn, "rb") as f:
        targets = extract_gesture_labels(f,label_tier_name,fps=10)

    # Train and test splits
    pose_data = torch.tensor(pose_data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    print(f"pose data shape: {pose_data.shape}")
    n = int(0.9 * len(pose_data))  # first 90% will be the train data, rest val data
    train_data, train_targets = pose_data[:n], targets[:n]
    val_data, val_targets = pose_data[n:], targets[n:]

    model = PoseBERT(pose_dim, model_dim).to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01)

    print("-----Start training pose BERT---------------")
    os.makedirs(OUT_DIR,exist_ok=True)
    train_loss_list=[]
    val_loss_list=[]
    best_loss = float('inf')
    for epoch in range(max_epochs):
        start_time = time.time()
        epoch_losses=[]
        print(f'--------Epoch: {epoch+1}--------------')
        for iter in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model, train_data, val_data)
                train_loss_list.append(losses['train'])
                val_loss_list.append(losses['val'])
                epoch_losses.append(losses['val'])
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = get_batch('train', train_data, val_data,train_targets,val_targets)

            # backpropagation
            logits, loss = model(xb,yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        print(f"------Epoch{epoch + 1} {(end_time - start_time) / 60} min spent-----")
        # save best model
        epoch_loss = sum(epoch_losses) / max_iters
        if epoch_loss < best_loss:
            torch.save(model.state_dict(), f"{OUT_DIR}/best.pth")
            best_loss = epoch_loss
        # save the model at the last training step
        torch.save(model.state_dict(), last_save_path)
        save_loss_plot(plt_save_path, train_loss_list, val_loss_list)

    print("-----poseBERT Successfully trained---------------")