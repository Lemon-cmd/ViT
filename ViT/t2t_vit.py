import torch 
import torch.nn as nn
import torch.nn.functional as F

from padder import Padder
from attention import Attention
from transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ReshapeImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(x.size(1) ** 0.5))

class T2TViT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, mlp_dim, 
                 dim_head=64, depth=1, heads=1, pos_len=1000, pool='cls', 
                 dropout=0.1, layer_dropout=0.0, t2t_layers=[(7, 4), (3, 2), (3, 2)], keepdim=False):

        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        for i, (size, stride) in enumerate(t2t_layers):
            assert(stride > 0)
            in_channels *= size ** 2

            layers.extend([
                ReshapeImage() if i > 0 else nn.Identity(),
                nn.Unfold(kernel_size = size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(in_channels, dim_head = in_channels, 
                    mlp_dim = in_channels, dropout = dropout) if i != len(t2t_layers) - 1 else nn.Identity()
            ])

        layers.append(nn.Linear(in_channels, hid_dim))

        self.pool = pool
        self.patch_drop = nn.Dropout(dropout)
        self.patch_embed = nn.Sequential(*layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len + 1, hid_dim))
        
        self.transformer = Transformer(hid_dim, dim_head, mlp_dim, depth, heads, layer_dropout)
                    
        self.mlp_head = nn.Sequential( 
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, out_channels)
        )

        if keepdim:
            self.reshape = ReshapeImage()
            
    def _pool(self, x):
        return x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

    def forward(self, x, return_seq = False):
        assert(len(x.shape) == 4)

        # project C x H x W through token2token layers 
        # i.e., a series of transformers and unfold layers
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
            return cls, hid_seq
        
        return self.mlp_head(self._pool(x)) 
