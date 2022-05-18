import torch 
import torch.nn as nn
import torch.nn.functional as F

from padder import Padder
from t2t_vit import ReshapeImage
from transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
   
class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, mlp_dim, 
                 dim_head=64, patch_size=1, depth=1, heads=1, pos_len=1000,
                 pool='cls', dropout=0.1, layer_dropout=0.1, channel_first=False, keepdim=False): 
        
        super().__init__()
        
        assert type(patch_size) in {tuple, int}, 'patch_size must be either an int or a tuple'
        assert pool in {'cls', 'mean'}, 'pool type must be either None, cls (cls token), or mean (mean pooling)'
            
        if type(patch_size) == tuple:
            assert(len(patch_size) == 2)
            d1, d2 = patch_size
        else:
            d1 = d2 = patch_size
        
        if channel_first:
            patcher = lambda : Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                         p1 = d1, p2 = d2, c = in_channels)
        else:
            patcher = lambda : Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', 
                                         p1 = d1, p2 = d2, c = in_channels)
                     
        self.pool = pool
        self.channel_first = channel_first
        self.patch_dim = d1 * d2 * in_channels
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.padder = Padder(d1, d2, channel_first = channel_first)
        
        self.patch_drop = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len + 1, hid_dim))
        self.patch_embed = nn.Sequential(patcher(), nn.Linear(self.patch_dim, hid_dim))
        self.transformer = Transformer(hid_dim, dim_head, mlp_dim, depth, heads, layer_dropout)
        
        d = (patch_size ** 2) ** 2 if keepdim else 1
            
        self.mlp_head = nn.Sequential( 
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, out_channels * d)
        )
        
        self.reshape = nn.Sequential(
            Rearrange('b n (c k) -> b (n k) c', c = out_channels),
            ReshapeImage()
        )

    def _pool(self, x):
        return x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
    def forward(self, x, return_seq = False):
        assert(len(x.shape) == 4)
        
        # pad input such that h % p1 = 0 and w % p2 = 0
        x = self.padder(x)

        # turn H x W x C into N x P^2 x C where N = HW / P^2
        x = self.patch_embed(x)

        # add CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        x = torch.cat((x, cls_tokens), dim = 1)
            
        x += self.pos_embed[:, :x.size(1)]
        x = self.patch_drop(x)
            
        # apply transformer
        x = self.transformer(x)
        
        if return_seq:
            x = self.mlp_head(x)
            cls, hid_seq = self._pool(x), x[:, 1:]
            return cls, self.reshape(hid_seq)
        
        return self.mlp_head(self._pool(x)) 
