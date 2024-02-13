import torch
import math
from torch import Tensor

class LogMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        assert input.ndim == 1
         #fixme: add mask
        x = (input - target).square()
        return torch.log(x).mean()
