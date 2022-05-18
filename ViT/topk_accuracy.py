import torch

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    assert(topk > 0)
    with torch.no_grad():
        yh = output.topk(k = topk, dim = -1)[1]
        y = target.unsqueeze(-1).repeat(1, yh.size(1))
        return (yh == y).float().sum().item() / y.size(0)   