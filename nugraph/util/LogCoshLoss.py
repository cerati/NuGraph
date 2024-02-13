import torch
import math
from torch import Tensor
import torch.nn.functional as F

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        assert input.ndim == 2
        mask = target[:,0]>0
        mask = torch.logical_and(mask,target[:,0]<255)
        mask = torch.logical_and(mask,target[:,1]>-116)
        mask = torch.logical_and(mask,target[:,1]<116)
        mask = torch.logical_and(mask,target[:,2]>10)
        mask = torch.logical_and(mask,target[:,2]<1036)
        x = (input[mask] - target[mask]).square().sum(dim=1).sqrt()
        return (x + F.softplus(-2. * x) - math.log(2.0)).mean()
