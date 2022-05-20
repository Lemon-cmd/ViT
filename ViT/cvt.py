import torch 
import torch.nn as nn
import torch.nn.functional as F

from t2t_vit import ReshapeImage
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ChannelNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        
        self.fn = fn
        self.norm = ChannelNorm(dim)
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    

class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias = True):
        super().__init__()
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, int), "Kernel size must be a tuple or an integer."

        self.fn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups = in_channels, bias = bias),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = bias)    
        )

    def forward(self, x):
        return self.fn(x)
    
class Attention(nn.Module):
    def __init__(self, dim, kernel_size, stride, hid_dim=64, heads=8, dropout=0.):
        super().__init__()
        
        self.heads = heads
        
        padding = kernel_size // 2
        self.q_proj = DepthWiseConv2d(dim, hid_dim * heads, kernel_size, 1, padding, bias = False)
        self.k_proj = DepthWiseConv2d(dim, hid_dim * heads, kernel_size, stride, padding, bias = False)
        self.v_proj = DepthWiseConv2d(dim, hid_dim * heads, kernel_size, stride, padding, bias = False)
        
        self.reshaper = ReshapeImage()
        self.dropout = nn.Dropout(dropout)
        self.split_head = Rearrange('b l (h d) -> (b h) l d', h = heads, d = hid_dim)
        
        self.o_proj = nn.Sequential(
            nn.Conv2d(hid_dim * heads, dim, kernel_size=1),
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
        return self.o_proj(self.reshaper(av))
    
    def forward(self, x):
        # M x W x (h x d)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'b (h c) x y -> b (x y) (h c)', h = self.heads)
        k = rearrange(k, 'b (h c) x y -> b (x y) (h c)', h = self.heads)
        v = rearrange(v, 'b (h c) x y -> b (x y) (h c)', h = self.heads)
        
        o = self._scaled_dot_prod(q, k, v)
        return o
    
    
class Transformer(nn.Module):
    def __init__(self, dim, kernel_size, stride, dim_head=64, heads=8, depth=1, mlp_mult=4, dropout=0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        attn_blk = lambda: PreNorm(dim, Attention(dim, kernel_size, stride, 
                                                  dim_head, heads, dropout))
        
        ffn_blk = lambda: PreNorm(dim, FeedForward(dim, mlp_mult, dropout))
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attn_blk(), ffn_blk()
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def get_layer(in_channels, emb_dim, emb_kernel, emb_stride, proj_kernel, kv_stride, heads, depth, mlp_mult, dropout):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, emb_dim, emb_kernel, padding = emb_kernel//2, stride = emb_stride),
        ChannelNorm(emb_dim),
        Transformer(emb_dim, proj_kernel, kv_stride, 
                    heads=heads, depth=depth, mlp_mult=mlp_mult, dropout=dropout)
    )
    
    return layer
    
class CvT(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 s1_emb_dim = 64, 
                 s1_emb_kernel = 7, 
                 s1_emb_stride = 4,
                 s1_proj_kernel = 3, 
                 s1_kv_stride = 2,
                 s1_heads = 1, 
                 s1_depth = 1, 
                 s1_mlp_mult = 4,
                 s2_emb_dim = 192,
                 s2_emb_kernel = 3,
                 s2_emb_stride = 2,
                 s2_proj_kernel = 3,
                 s2_kv_stride = 2,
                 s2_heads = 3,
                 s2_depth = 2,
                 s2_mlp_mult = 4,
                 s3_emb_dim = 384,
                 s3_emb_kernel = 3,
                 s3_emb_stride = 2,
                 s3_proj_kernel = 3,
                 s3_kv_stride = 2,
                 s3_heads = 6,
                 s3_depth = 10,
                 s3_mlp_mult = 4,
                 dropout = 0.0):
        
        super().__init__()
        
        layers = []
        layers.append(get_layer(
            in_channels, s1_emb_dim, s1_emb_kernel, s1_emb_stride, 
            s1_proj_kernel, s1_kv_stride, s1_heads, s1_depth, s1_mlp_mult, dropout)
        )
        
        layers.append(get_layer(
            s1_emb_dim, s2_emb_dim, s2_emb_kernel, s2_emb_stride, 
            s2_proj_kernel, s2_kv_stride, s1_heads, s2_depth, s2_mlp_mult, dropout)
        )
        
        layers.append(get_layer(
            s2_emb_dim, s3_emb_dim, s3_emb_kernel, s3_emb_stride, 
            s3_proj_kernel, s3_kv_stride, s3_heads, s3_depth, s3_mlp_mult, dropout)
        )
        
        self.layers = nn.Sequential(*layers)
        
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(s3_emb_dim, out_channels)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return self.mlp_head(x).unsqueeze(1)