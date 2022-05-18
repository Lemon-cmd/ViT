import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class Transformer(nn.Module):
    def __init__(self, dim, dim_head, mlp_dim, depth=1, heads=1, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        attn_blk = lambda : PreNorm(dim, Attention(dim, heads = heads, 
                                    hid_dim = dim_head, dropout = dropout))
        
        ffn_blk = lambda : PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([attn_blk(), ffn_blk()]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
