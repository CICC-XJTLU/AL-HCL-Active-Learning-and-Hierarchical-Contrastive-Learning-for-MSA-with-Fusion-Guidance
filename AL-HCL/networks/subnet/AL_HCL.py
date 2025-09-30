import torch
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *
import math
# from .HCL_Module import CL3, HCLModule_3, HCLModule

class AL_HCL(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2, name=""):
        super(AL_HCL, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.cat_connect = nn.Linear(3 * TEXT_DIM, TEXT_DIM)
        self.cross_ATT_visual = CrossAttention(dim=TEXT_DIM)
        self.cross_ATT_acoustic = CrossAttention(dim=TEXT_DIM)
        self.layer_norm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(TEXT_DIM, TEXT_DIM, batch_first=True)
        self.fc1 = nn.Linear(TEXT_DIM, hidden_size)
        self.fc2 = nn.Linear(hidden_size, TEXT_DIM)
        self.cat_connect_two = nn.Linear(2 * TEXT_DIM, TEXT_DIM)

    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        # Extract visual and acoustic feature
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)

        # Concatenate and transform features
        fusion = torch.cat((text_embedding, visual_, acoustic_), dim=2)
        fusion = self.cat_connect(fusion)

        visual_ = self.cross_ATT_acoustic(fusion, visual_, visual_)
        acoustic_ = self.cross_ATT_acoustic(fusion, acoustic_, acoustic_)

        # Concatenate all features for LSTM input
        shift = torch.cat((text_embedding, visual_, acoustic_), dim=2)
        shift = self.cat_connect(shift)
        shift = self.layer_norm(shift + text_embedding)  # Residual connection and layer normalization

        # Final output from LSTM
        output, _ = self.lstm(shift)  # Ensure output is a tensor
        output += text_embedding
        
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_per_head = dim // heads
        self.scale = self.dim_per_head ** -0.5

        assert dim % heads == 0, "Dimension must be divisible by the number of heads."

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, d = queries.size()
        h = self.heads

        queries = self.query(queries).view(b, n, h, self.dim_per_head).transpose(1, 2)
        keys = self.key(keys).view(b, -1, h, self.dim_per_head).transpose(1, 2) # Fix here
        values = self.value(values).view(b, -1, h, self.dim_per_head).transpose(1, 2) # Fix here

        dots = torch.einsum('bhqd,bhkd->bhqk', queries, keys) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, h, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhqk,bhvd->bhqd', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        # 确保 out 是 Tensor
        return out  # 返回的是 Tensor
