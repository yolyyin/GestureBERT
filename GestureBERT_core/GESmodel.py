import torch.nn as nn
from BERTembedding import BERTEmbedding

PAD_IDX= -3.0
MASK_IDX= 2.0
UNK_IDX = -2.0


class GesModel(nn.Module):
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
        self.criterion = nn.CrossEntropyLoss()

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
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, labels):
        # attention masking for padded token
        # x shape [batch_size,seq_len,pose_embed_size]
        # src_key_padding_mask mask size: torch.ByteTensor([batch_size, seq_len)
        padding_mask = (x[:,:,0] == PAD_IDX)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer encoder blocks
        x = self.encoder(x,src_key_padding_mask=padding_mask)

        x = self.head(x)
        logits_flat = x.view(-1, 2)  # [b*t,2]
        labels_flat = labels.view(-1)  # [b*t]
        loss = self.criterion(logits_flat, labels_flat)
        return loss, x



