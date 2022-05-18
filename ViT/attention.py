import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim, hid_dim, heads=8, dropout=0.1, **kwargs):
        super().__init__()

        self.heads = heads
        self.q_proj = nn.Linear(dim, hid_dim * heads, False)
        self.k_proj = nn.Linear(dim, hid_dim * heads, False)
        self.v_proj = nn.Linear(dim, hid_dim * heads, False)

        self.dropout = nn.Dropout(dropout)

        self.split_head = Rearrange('b l (h d) -> (b h) l d', h = heads, d = hid_dim)

        self.o_proj = nn.Sequential(
            nn.Linear(hid_dim * heads, dim),
            nn.Dropout(dropout)
        )
        
    def _scaled_dot_prod(self, q, k, v):
        shape = q.shape
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        # (M x H) x L x L
        a = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** (0.5))
        a = torch.softmax(a, dim = -1)
        a = self.dropout(a)
        
        # v : (M x H) x L x dk
        av = torch.bmm(a, v)
       
        # M x L x (H x dk)
        av = av.reshape(shape)

        # o : M x L x d
        return self.o_proj(av)
                
    def forward(self, x, **kwargs):
        # M x W x (h x d)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        o = self._scaled_dot_prod(q, k, v)
        return o   
