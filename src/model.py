#based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, emb_size]
        Returns:
            Tensor of same shape
        """
        x = x + self.pe[ : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, in_dim:int=2, out_dim:int=1, d_hid:int=512, nlayers:int=3,
            emb_size:int=512, nhead:int=8, dropout:float=0.5, tm_classifier:bool=False,
            max_length:int=None):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_size, dropout, max_length)
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(in_dim, emb_size)
        self.emb_size = emb_size
        self.decoder = nn.Linear(emb_size, out_dim)

        #input: mean(encoding(sequence)), encoding(last point)
        #output: velocity of next point
        self.v_predictor = nn.Linear(emb_size*2, 1)
        # input: mean(encoding(sequence)), encoding(last point)
        # output: cos(𝛼), sin(𝛼) where 𝛼 is angle of next point
        self.cos_sin_layer = nn.Linear(emb_size*2, 2)
        self.cos_sin_act = nn.Tanh()

        self.tm_classifier = tm_classifier
        self.store_norm_vals(torch.scalar_tensor(1),
            torch.zeros(in_dim-2), torch.ones(in_dim-2)) #init values

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    @property
    def device(self):
        """src: https://stackoverflow.com/a/70593357/11227118 """
        return next(self.parameters()).device


    def encode(self, src:Tensor, src_mask:Tensor=None) -> Tensor:
        out = self.encoder(src) * math.sqrt(self.emb_size)
        out = self.pos_encoder(out)
        return self.transformer_encoder(out, src_mask)

    def decode(self, encodings:Tensor) -> Tensor:
        out = self.decoder(encodings)
        if not self.tm_classifier:
            out = torch.sigmoid(out)
        return out

    def cos_sin_predictor(self, src):
        return self.cos_sin_act(self.cos_sin_layer(src))


    def freeze_encoder(self, unfreezed_layers:int=0):
        """
        Can be used for transfer learning
        unfreezed_layers: Number of layers at the end of encoder,
        which should stay unfreezed
        """
        n_layers = len(self.transformer_encoder.layers)

        assert type(unfreezed_layers) == int \
            and unfreezed_layers >= 0 and unfreezed_layers <= n_layers, \
                "number of unfreezed_layers needs to be positive integer between 0 and nlayers"

        freezed_layers = n_layers - unfreezed_layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.layers[:freezed_layers].parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True


    def forward(self, src:Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, in_dim]

        Returns:
            out:                Tensor of shape [batch_size, seq_len, 1]
            cos_sin_next_point: Tensor of shape [batch_size, 2]
            v_next_point:       Tensor of shape [batch_size, 1]
        """
        encodings = self.encode(src)
        seq_encod_mean = encodings.mean(-2) # average encoding over whole sequence
        out = self.decode(encodings)

        # context: mean(encoding(sequence)), encoding(last point)
        context = torch.cat((seq_encod_mean, encodings[...,-1,:]), dim=-1)

        return out, self.cos_sin_predictor(context), self.v_predictor(context)

    def store_norm_vals(self, coords_std, feat_mean, feat_std):
        """Store norm values for transfer learning"""
        self.register_buffer('coords_std', coords_std)
        self.register_buffer('feat_mean', feat_mean)
        self.register_buffer('feat_std', feat_std)