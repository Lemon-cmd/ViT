from requests import patch
import torch 
import torch.nn as nn
import torch.nn.functional as F

from padder import Padder
from t2t_vit import ReshapeImage
from transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias = True):
        super().__init__()
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, int), "Kernel size must be a tuple or an integer."

        self.fn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups = in_channels, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, bias = bias, padding='same')    
        )

    def forward(self, x):
        return self.fn(x)


class Pool(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        
        self.reshaper = ReshapeImage()
        self.cls_proj = nn.Linear(dim, dim * 2)
        self.down_sample = DepthWiseConv2d(dim, dim * 2, kernel_size, stride, padding)
    
    def forward(self, x):
        cls, hid_seq = x[:, :1], x[:, 1:]
        
        cls = self.cls_proj(cls)
        hid_seq = self.reshaper(hid_seq) 
        hid_seq = self.down_sample(hid_seq)
        hid_seq = rearrange(hid_seq, 'b c h w -> b (h w) c')

        return torch.cat((cls, hid_seq), dim = 1)
        
        
class PiT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, mlp_dim, 
                 depth=((3, 3, 32, 5, 1, 1), (3, 3, 32, 5, 1, 1), (3, 3, 32, 5, 1, 1)), 
                 patch_size=1, pos_len=1000, pool='cls', dropout=0.1, layer_dropout=0.0, keepdim=False): 
        
        super().__init__()
        assert isinstance(depth, tuple)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool
        patch_dim = (patch_size ** 2) * in_channels
        
        patch_stride = patch_size // 2
        patch_stride = 1 if patch_stride == 0 else patch_stride
        
        self.patch_embed = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, hid_dim),
        )

        self.patch_drop = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len + 1, hid_dim))

        layers = []
        for i in range (len(depth)):
            depths, heads, dim_head, kernel_size, stride, padding = depth[i]
            
            layers.append(
                Transformer(hid_dim, dim_head, mlp_dim,
                    depth=depths, heads=heads, dropout=layer_dropout),        
            )

            if i != len(depth) - 1:
                layers.append(Pool(hid_dim, kernel_size, stride, padding))
                hid_dim *= 2

        self.transformer = nn.Sequential(*layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, out_channels)
        )
        
        if keepdim:
            self.reshape = nn.Sequential(
                ReshapeImage()
            )
        
    def _pool(self, x):
        return x.mean(dim = 1) if self.pool == 'mean' else x[:, :1]

    def forward(self, x, return_seq=False):
        assert(len(x.shape) == 4)            
        x = self.patch_embed(x)

        # add CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        x = torch.cat((x, cls_tokens), dim = 1)

        x += self.pos_embed[:, :x.size(1)]
        x = self.patch_drop(x)

        # perform PiT
        x = self.transformer(x)

        if return_seq:
            x = self.mlp_head(x)
            cls, hid_seq = self._pool(x), x[:, 1:]
            return cls, self.reshape(hid_seq)
        
        return self.mlp_head(self._pool(x)) 