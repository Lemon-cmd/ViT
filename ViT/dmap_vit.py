from re import S
from requests import patch
import torch, math
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from padder import Padder
from t2t_vit import ReshapeImage
from cvt import ChannelNorm, FeedForward

from pit import PiT
from cvt import CvT
from transformer import Transformer

class Patcher(nn.Module):
    def __init__(self, dim, hid_dim, kernel_size=3, stride=1, padding='same', dilation=1) -> None:
        super().__init__()        
        assert hid_dim > 1, 'parameter needs to be greater than 1'
        
        dmodel = hid_dim // 2
        self._fn = nn.Sequential(
            ChannelNorm(dim),
            nn.Conv2d(dim, dmodel, kernel_size, stride, padding, dilation),
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(dmodel, hid_dim),
        )
        
    def forward(self, x) -> torch.Tensor: return self._fn(x)  

class CrossProdMean(nn.Module):
    def __init__(self, dim, hid_dim) -> None:
        super().__init__()
    
        self._proj_a = nn.Linear(dim, hid_dim)
        self._proj_b = nn.Linear(dim, hid_dim)
        self._g = nn.Parameter(torch.randn(hid_dim))
        
    def forward(self, x) -> torch.Tensor:
        a = self._proj_a(x)
        b = self._proj_b(x)
        
        o = self._g * torch.einsum('bni, bnj -> bnij', a, b).mean(-1)
        return o

class Partitioner(nn.Module):
    def __init__(self, dim, hid_dim, mlp_dim, depth=1, heads=8, dim_head=64, patch_size=16, blk_size=1000, dropout=0.1) -> None:
        super().__init__()
        assert (math.isqrt(blk_size))
        
        self._crossprod = CrossProdMean(dim, hid_dim)
        
        self._patcher = nn.Sequential(
            ReshapeImage(),
            ChannelNorm(hid_dim),
            Padder(patch_height=patch_size, patch_width=patch_size, channel_first=True),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(hid_dim * (patch_size ** 2), hid_dim),
            Transformer(hid_dim, dim_head, mlp_dim, depth, heads, dropout),
        )
        
        self._pos_embed = nn.Parameter(torch.randn(1, blk_size, hid_dim))
        
        self._reshape = nn.Sequential(
            nn.Linear(hid_dim, dim * (patch_size ** 2)),
            Rearrange('b n (c p) -> b (n p) c', p = patch_size ** 2)   
        )

    def forward(self, x) -> torch.Tensor:
        x = self._crossprod(x)
        crops = self._patcher(x)
        crops += self._pos_embed[:, : crops.size(1)]
        return self._reshape(crops)
    
class DmapViT(nn.Module):
    def  __init__(self, in_channels, out_channels, embed_dim, hid_dim, mlp_dim, 
                  depth=4, heads=8, dim_head=64, patch_size=32, vit_heads=4, blk_size=1024, 
                  pool='cls', patch_dropout=0.1, layer_dropout=0.1, vit=None) -> None:
        
        super().__init__()
        assert math.isqrt(blk_size), 'Block size must be a perfect square.'
        assert pool in {'cls', 'mean'}, 'Pool type must be either None, cls (cls token), or mean (mean pooling).'
        
        self._blk_size = blk_size
        self._patch_proj = Patcher(in_channels, embed_dim)
        self._partitioner = Partitioner(embed_dim, hid_dim, mlp_dim, 
                                        depth=depth, heads=heads, dim_head=dim_head,
                                        dropout=patch_dropout, blk_size=blk_size, patch_size=patch_size)
        
        if vit == None:
            self._cls_token = PiT(embed_dim, 
                                  embed_dim * vit_heads, hid_dim, mlp_dim, 
                                  depth=((6, 8, 64, 3, 2, 1), (4, 4, 32, 2, 2, 1), (2, 4, 32, 3, 1, 1)), 
                                  patch_size=patch_size, pos_len=blk_size, pool='cls', dropout=patch_dropout, layer_dropout=layer_dropout, keepdim=False)
        else:
            self._cls_token = vit 
        
        self._vh_proj = nn.Parameter(torch.randn(vit_heads))
        
        self._cls_token = nn.Sequential(
            ReshapeImage(),
            self._cls_token,
            Rearrange('b n (c h) -> b n c h', c = embed_dim, h = vit_heads)
        )    

        self.o_reshape = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            ReshapeImage()
        )
        
    def forward(self, x) -> torch.Tensor:
        m = x.size(2) * x.size(3)
        x = self._patch_proj(x)
        
        cls = self._cls_token(x).squeeze(1)
        crops = self._partitioner(x)[:, : m]
        
        cls = (cls @ self._vh_proj).unsqueeze(1)
        return self.o_reshape(cls + crops)
    
def main():
    model = DmapViT(in_channels=124, out_channels=2, embed_dim=2, hid_dim=2, mlp_dim=2, 
                    depth=12, heads=12, dim_head=64, patch_size=16, vit_heads=16, blk_size=1024, 
                    pool='cls', patch_dropout=0.2, layer_dropout=0.1, \
                        
                    vit = CvT(
                            in_channels=2, out_channels=2 * 16, 
                            
                            s1_emb_dim = 64, s1_emb_kernel = 7, s1_emb_stride = 4, s1_proj_kernel = 3, 
                            s1_kv_stride = 2, s1_heads = 8, s1_depth = 4, s1_mlp_mult = 4,
                            
                            s2_emb_dim = 128, s2_emb_kernel = 3, s2_emb_stride = 2, s2_proj_kernel = 3,
                            s2_kv_stride = 2, s2_heads = 12, s2_depth = 6, s2_mlp_mult = 4, 
                            
                            s3_emb_dim = 256, s3_emb_kernel = 3, s3_emb_stride = 2, s3_proj_kernel = 3, s3_kv_stride = 2,
                            s3_heads = 16, s3_depth = 8, s3_mlp_mult = 2,
                            
                            dropout = 0.1
                        )
    )
    
    x = torch.randn(2, 124, 128, 128)
    
    print(model(x).shape)
    
main()
    
                            