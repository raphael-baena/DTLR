
import torch
import torch.nn as nn
import math
from util.misc import inverse_sigmoid


def positional_encoding(tensor, d_model):
    dim_range = torch.arange(d_model, dtype=torch.float32) // 2
    dim_range = dim_range.unsqueeze(0).unsqueeze(1).cuda()
    scale = 2 * math.pi
    exponents = 2 * dim_range / d_model
    angles = scale * tensor.unsqueeze(-1) / (10000 ** exponents)
    sin_vals = torch.sin(angles[:, :, 0::2])
    cos_vals = torch.cos(angles[:, :, 1::2])

    position_encodings = torch.cat([sin_vals, cos_vals], dim=-1)
    return position_encodings
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class SelfAttentionModel(nn.Module):
    def __init__(self, input_size, d_model, output_size, n_heads):
        super(SelfAttentionModel, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.fc1 = nn.Linear(input_size, d_model)

        self.self_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
    
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn3 = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn4 = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm4 = nn.LayerNorm(d_model)
        self.ff = PositionWiseFeedForward(d_model, 2048)
        self.dropoutff = nn.Dropout(0.1)
        self.normff = nn.LayerNorm(d_model)

        self.fc2 = nn.Linear(d_model, output_size)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            print(p)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,b):
        b_pos = b[:,:,0] + b[:,:,2]//2
        pos_encoding = positional_encoding(b_pos, self.d_model).cuda()#[:,:,0].unsqueeze(-1)  
        x = self.fc1(x)
        tgt = x + pos_encoding
        
        q = k = tgt.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        # q = torch.zeros_like(x).permute(1, 0, 2)
        # q = self.fc1(pos_encoding).permute(1, 0, 2)
        # k = x
        
        x2 = self.self_attn1(q, k, x)[0]

        x2 = x2.permute(1, 0, 2)
        x = x.permute(1, 0, 2)

        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # preds.append(self.fc2(x))
        tgt = x + positional_encoding(b_pos, x.shape[-1]).cuda()[:,:,0].unsqueeze(-1)
        q = k = tgt.permute(1, 0, 2)
        x  = x.permute(1, 0, 2)

        x2 = self.self_attn2(q, k, x)[0]

        x2 = x2.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + self.dropout2(x2)
        x = self.norm2(x)


        tgt = x + positional_encoding(b_pos, x.shape[-1]).cuda()[:,:,0].unsqueeze(-1)
        q = k = tgt.permute(1, 0, 2)
        x  = x.permute(1, 0, 2)

        x2 = self.self_attn3(q, k, x)[0]

        x2 = x2.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + self.dropout3(x2)
        x = self.norm3(x)

        # preds.append(self.fc2(x))
        tgt = x + positional_encoding(b_pos, x.shape[-1]).cuda()[:,:,0].unsqueeze(-1)
        q = k = tgt.permute(1, 0, 2)
        x  = x.permute(1, 0, 2)

        x2 = self.self_attn4(q, k, x)[0]

        x2 = x2.permute(1, 0, 2)
        x = x.permute(1, 0, 2)

        x = x + self.dropout4(x2)
        x = self.norm4(x)


        ff_output = self.ff(x)
        x = x + self.dropoutff(ff_output)
        x  = self.normff(x)

        x = self.fc2(x)
        # preds.append(x)
        return x
    
    