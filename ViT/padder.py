import torch 
import torch.nn as nn
import torch.nn.functional as F

class Padder(nn.Module):
    def __init__(self, patch_height, patch_width, channel_first = False):
        super().__init__()
        
        self.d1 = patch_height
        self.d2 = patch_width
        self.channel_first = channel_first
        
    @staticmethod
    def _forward_clast(x, d1, d2):
        m, h, w, c = x.shape

        if w % d2 > 0:
            d = w + d2
            d = d + (d2 - d % d2)
            x = F.pad(x, (0, 0, 0, d - w), 'constant', 1e-8)
        
        if h % d1 > 0:
            d = h + d1
            d = d + (d1 - d % d1)
            x = F.pad(x, (0, 0, 0, 0, 0, d - h), 'constant', 1e-8)
        
        return x
    
    @staticmethod
    def _forward_cfirst(x, d1, d2):
        m, c, h, w = x.shape
        
        if w % d2 > 0:
            d = w + d2
            d = d + (d2 - d % d2)
            x = F.pad(x, (0, d - w), 'constant', 1e-8)
        
        if h % d1 > 0:
            d = h + d1
            d = d + (d1 - d % d1)
            x = F.pad(x, (0, 0, 0, d - h), 'constant', 1e-8)
            
        return x
        
    def forward(self, x):        
        if self.channel_first:
            return Padder._forward_cfirst(x, self.d1, self.d2)
        return Padder._forward_clast(x, self.d1, self.d2)