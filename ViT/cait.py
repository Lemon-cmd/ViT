import torch 
import torch.nn as nn
import torch.nn.functional as F

from padder import Padder
from t2t_vit import ReshapeImage
from attention import Attention as SAttention
from transformer import PreNorm, FeedForward

from random import randrange
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers
            
    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.scale * self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, hid_dim, heads=8, dropout=0.1, **kwargs):
        super().__init__()

        self.heads = heads
        self.q_proj = nn.Linear(dim, hid_dim * heads, False)
        self.k_proj = nn.Linear(dim, hid_dim * heads, False)
        self.v_proj = nn.Linear(dim, hid_dim * heads, False)

        self.dropout = nn.Dropout(dropout)
        self.pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.post_attn = nn.Parameter(torch.randn(heads, heads))
        
        self.split_head = Rearrange('b l (h d) -> (b h) l d', h = heads, d = hid_dim)

        self.o_proj = nn.Sequential(
            nn.Linear(hid_dim * heads, dim),
            nn.Dropout(dropout)
        )

    def _mix_attn(self, a):
        # (M x H) x L x L
        shape = a.shape
        
        a = rearrange(a, '(m h) i j -> m h i j', 
                h = self.heads, i = a.size(1), j = a.size(2))

        a = torch.einsum('m h i j, h g -> m g i j', a, self.pre_attn)
        a = torch.softmax(a, dim = -1)
        a = self.dropout(a)

        a = torch.einsum('m h i j, h g -> m g i j', a, self.post_attn)
        return a.reshape(shape)
                        
    def _scaled_dot_prod(self, q, k, v):
        shape = q.shape
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        # (M x H) x L x L
        a = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** (0.5))

        # perform mxi attention
        a = self._mix_attn(a)
        
        # v : (M x H) x L x dk
        av = torch.bmm(a, v)
       
        # M x L x (H x dk)
        av = av.reshape(shape)

        # o : M x L x d
        return self.o_proj(av)

    def forward(self, x, context=None):
        context = x if context == None else torch.cat((x, context), dim = 1)

        # M x W x (h x d)
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        o = self._scaled_dot_prod(q, k, v)
        return o   

class Transformer(nn.Module):
    def __init__(self, dim, dim_head, mlp_dim, depth=1, heads=1, dropout=0.1, cls=False):
        super().__init__()
        
        self.layer_dropout = dropout
        self.layers = nn.ModuleList([])

        if cls:
            attn_blk = lambda : PreNorm(dim, Attention(dim, heads = heads, 
                                        hid_dim = dim_head, dropout = dropout))
        else:
            attn_blk = lambda : PreNorm(dim, SAttention(dim, heads = heads, 
                                        hid_dim = dim_head, dropout = dropout))

        ffn_blk = lambda : PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        
        for i in range(1, depth + 1):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, attn_blk(), depth = i), 
                LayerScale(dim, ffn_blk(), depth = i)
            ]))
            
    def forward(self, x, context=None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x


class CaiT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, mlp_dim, 
                 dim_head=64, patch_size=16, depth=1, cls_depth=1, heads=1, pos_len=1000,
                 pool = 'cls', dropout=0.1, layer_dropout=0., channel_first=False, keepdim=False): 
 
        super().__init__()

        assert type(patch_size) in {tuple, int}, 'patch_size must be either an int or a tuple'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
            
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
        self.padder = Padder(d1, d2, channel_first = channel_first)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len, hid_dim))
        
        self.patch_drop = nn.Dropout(dropout)
        self.patch_embed = nn.Sequential(patcher(), nn.Linear(self.patch_dim, hid_dim))

        self.patch_transformer = Transformer(hid_dim, dim_head, mlp_dim, depth, heads, layer_dropout)
        self.cls_transformer = Transformer(hid_dim, dim_head, mlp_dim, cls_depth, heads, layer_dropout, cls = True)
 
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

        # add position embedding
        x += self.pos_embed[:, :x.size(1)]
        x = self.patch_drop(x)

        # apply transformer for patch
        x = self.patch_transformer(x)
         
        # apply transformer for cls
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        x = self.cls_transformer(cls_tokens, context = x)
        
        if return_seq:
            x = self.mlp_head(x)
            cls, hid_seq = self._pool(x), x[:, 1:]
            return cls, self.reshape(hid_seq)
        
        return self.mlp_head(self._pool(x)) 
