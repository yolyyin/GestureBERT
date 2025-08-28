import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # [1,max_len,d_model]
        #self.register_buffer('pe', pe)
        self.pe = nn.Parameter(torch.randn(1,max_len, d_model))

    def forward(self, x):
        return self.pe[:, :x.size(1)] # [1,t,d_model]


class TokenEmbedding(nn.Module):
    def __init__(self, pose_embed_size, embed_size=512):
        super().__init__()
        self.embedding_dim = embed_size
        self.linear = nn.Linear(pose_embed_size, embed_size)

    def forward(self, x):
        return self.linear(x) #[b,t,embed_size]


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, pose_embed_size, embed_size, dropout=0.1):
        """
        :param pose_embed_size: pose embedding size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(pose_embed_size=pose_embed_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x) #[b,t,embed_size]