import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss

from BERTembedding import BERTEmbedding

PAD_IDX= -3.0
MASK_IDX= 2.0
UNK_IDX = -2.0


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, pose_embed_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param pose_embed_size: pose embedding size
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(pose_embed_size=pose_embed_size, embed_size=hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attn_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # attention masking for padded token
        # x shape [batch_size,seq_len,pose_embed_size]
        # src_key_padding_mask mask size: torch.ByteTensor([batch_size, seq_len)
        padding_mask = (x[:,:,0] == PAD_IDX)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer encoder blocks
        x = self.encoder(x,src_key_padding_mask=padding_mask)
        return x


class BERTpretrain(nn.Module):
    def __init__(self,bert:BERT,pose_embed_size,pose_mask_size,alpha=1.0,beta=0.4,gamma=0.5):
        super(BERTpretrain, self).__init__()
        self.bert = bert
        self.pose_predictor = MaskedPoseModel(self.bert.hidden,pose_embed_size)
        self.mask_predictor = MaskClassifier(self.bert.hidden,pose_mask_size)
        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self,x,labels,joint_masks):
        bert_embed=self.bert(x)# [b,t,hidden]
        x = self.pose_predictor(bert_embed)# [b,t,pose_embedding]
        # apply loss mask to avoid calculating unmasked and padding frames
        loss_mask = (labels > PAD_IDX).to(x.dtype).to(x.device) #[b,t,pose_embeding]
        x_masked= x*loss_mask
        labels_masked=labels*loss_mask
        pose_loss = F.mse_loss(x_masked,labels_masked)
        smooth_loss=F.l1_loss(x_masked[:,1:,:]-x_masked[:,:-1,:],labels_masked[:,1:,:]-labels_masked[:,:-1,:])

        predicted_joint_masks = self.mask_predictor(self.bert(x)) # [b,t,pose_mask_size]
        # apply loss mask to avoid calculating masked frames
        pose_mask_loss =self.binary_criterion(predicted_joint_masks,joint_masks.float())

        loss = self.alpha*pose_loss + self.beta*pose_mask_loss + self.gamma*smooth_loss

        return loss, pose_loss,pose_mask_loss,bert_embed


class MaskedPoseModel(nn.Module):
    def __init__(self, hidden, pose_embed_size):
        super(MaskedPoseModel, self).__init__()
        #self.linear = nn.Linear(hidden,pose_embed_size)


        self.linear1 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.norm= nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, pose_embed_size)


    def forward(self,x):
        #x = self.linear(x)


        x=self.relu(self.linear1(x))
        x=self.norm(x)
        x=self.linear2(x)

        return x


class MaskClassifier(nn.Module):
    """
    a classifier head to predict joint angle unknown mask
    """
    def __init__(self,hidden,pose_mask_size):
        super(MaskClassifier,self).__init__()
        #self.linear = nn.Linear(hidden,pose_mask_size)


        self.linear1 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, pose_mask_size)


    def forward(self,x):
        #x = self.linear(x)


        x = self.relu(self.linear1(x))
        x = self.norm(x)
        x = self.linear2(x)

        return x


class GesGroupClassifer(nn.Module):
    """
    a classifier head to predict if this frame belongs to a gesture group
    """
    def __init__(self,hidden,output):
        super(GesGroupClassifer,self).__init__()
        self.linear1 = nn.Linear(hidden,hidden)
        self.linear2 = nn.Linear(hidden,output)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class GesGroupModel(nn.Module):
    def __init__(self,bert:BERT):
        super(GesGroupModel, self).__init__()
        self.bert = bert
        #self.group_predictor = GesGroupClassifer(self.bert.hidden,output=5)
        #project the CLS token to 5 gesture type: iconic,symbolic,deictic,beat,other
        self.group_predictor= nn.Linear(self.bert.hidden, out_features=5)
        self.criterion=nn.CrossEntropyLoss()

    def forward(self,x,labels):
        x= self.bert(x)# [b,t,hidden]
        cls_vector=x[:,0,:] #[b,hidden]
        logits = self.group_predictor(cls_vector)# [b,5]
        # shape of labels: [b,t], notice that the in BERTdataset ges_labels are pre-padded and post-padded,
        # thus the index [1] of length sequence is the real ges type value
        labels_single=labels[:,1] #[b] # label for every frame should be same under current setting
        #logits_flat = x.view(-1,2) # [b*t,2]
        #labels_flat=labels.view(-1) # [b*t]
        #loss = self.criterion(logits_flat,labels_flat)
        loss=self.criterion(logits,labels_single)
        return loss,logits


