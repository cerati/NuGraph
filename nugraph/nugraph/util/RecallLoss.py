# as described in https://arxiv.org/abs/2106.14917

#import torch
#from torch import Tensor
#import torch.nn.functional as F
#from torchmetrics.functional import recall
#
#class RecallLoss(torch.nn.Module):
#    def __init__(self, ignore_index: int = -1):
#        super().__init__()
#        self.ignore_index = ignore_index
#
#    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#        # Temporarily in your model
#        #loss = F.cross_entropy(input, target, ignore_index=self.ignore_index)
#        target = target.clone()
#        target[target > 4] = -1
#        weight = 1 - recall(input, target, 'multiclass',
#                            num_classes=input.size(1),
#                            average='none',
#                            ignore_index=self.ignore_index)
#        ce = F.cross_entropy(input, target, reduction='none',
#                             ignore_index=self.ignore_index)
#        loss = weight[target] * ce
#        return loss.mean()

#from torchmetrics import Recall
#class RecallLoss(torch.nn.Module):
#    def __init__(self, num_classes: int, ignore_index: int = -1):
#        super().__init__()
#        self.ignore_index = ignore_index
#        # dist_sync_on_step=True syncs across ranks before returning
#        self.recall_metric = Recall(
#            task='multiclass',
#            num_classes=num_classes,
#            average='none',
#            ignore_index=ignore_index,
#            dist_sync_on_step=True  # <-- the critical flag
#        )#
#
#    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#        target = target.clone()
#        target[target > 4] = -1
#
#        # Syncs TP/total counts across all ranks before computing recall
#        weight = 1 - self.recall_metric(input, target)
#        self.recall_metric.reset()
#
#        #ce = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
#        #loss = weight[target] * ce
#        #return loss.mean()
#        mask = target != self.ignore_index
#        ce = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
#        loss = torch.where(mask, weight[target.clamp(min=0)] * ce, torch.zeros_like(ce))
#        return loss.mean()

#import torch
#import torch.distributed as dist
#from torch import Tensor
#import torch.nn.functional as F
#from torchmetrics.functional import recall

#class RecallLoss(torch.nn.Module):
#    def __init__(self, ignore_index: int = -1):
#        super().__init__()
#        self.ignore_index = ignore_index
#
#    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#        target = target.clone()
#        target[target > 4] = -1
#
#        weight = 1 - recall(input, target, 'multiclass',
#                            num_classes=input.size(1),
#                            average='none',
#                            ignore_index=self.ignore_index)
#
#        # Manually sync weights across all ranks
#        if dist.is_available() and dist.is_initialized():
#            dist.all_reduce(weight, op=dist.ReduceOp.AVG)
#
#        ce = F.cross_entropy(input, target, reduction='none',
#                             ignore_index=self.ignore_index)
#        mask = target != self.ignore_index
#        loss = torch.where(mask, weight[target.clamp(min=0)] * ce,
#                           torch.zeros_like(ce))
#        return loss.mean()

#class RecallLoss(torch.nn.Module):
#    def __init__(self, num_classes: int, ignore_index: int = -1, momentum: float = 0.1):
#        super().__init__()
#        self.ignore_index = ignore_index
#        self.momentum = momentum
#        self.register_buffer('running_recall', torch.ones(num_classes))
#
#    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#        target = target.clone()
#        target[target > 4] = -1
#
#        # Compute recall on current batch
#        batch_recall = recall(input, target, 'multiclass',
#                              num_classes=input.size(1),
#                              average='none',
#                              ignore_index=self.ignore_index)
#
#        # Sync across GPUs
#        if dist.is_available() and dist.is_initialized():
#            dist.all_reduce(batch_recall, op=dist.ReduceOp.AVG)
#
#        # Update running estimate with momentum (like BatchNorm does)
#        self.running_recall = (1 - self.momentum) * self.running_recall \
#                              + self.momentum * batch_recall
#
#        weight = 1 - self.running_recall
#
#        ce = F.cross_entropy(input, target, reduction='none',
#                             ignore_index=self.ignore_index)
#        mask = target != self.ignore_index
#        loss = torch.where(mask, weight[target.clamp(min=0)] * ce,
#                           torch.zeros_like(ce))
#        return loss.mean()

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.classification import MulticlassRecall
import torch.distributed as dist

class RecallLoss(torch.nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.recall_metric = MulticlassRecall(
            num_classes=num_classes,
            average='none',
            ignore_index=ignore_index,
            #zero_division=1.0,  # undefined recall → recall=1.0 → weight=0.0 (ignore)
            sync_on_compute=True  # syncs TP/FN counts before compute()
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.clone()
        target[target > 4] = -1

        # Ensure metric is on same device as input
        self.recall_metric = self.recall_metric.to(input.device)

        self.recall_metric.update(input, target)
        weight = 1 - self.recall_metric.compute()  # globally correct recall

        ## Diagnostic: write weights from each rank to a file
        #if dist.is_available() and dist.is_initialized():
        #    rank = dist.get_rank()
        #else:
        #    rank = 0
        #import os
        #with open(f"/global/u1/c/cerati/NuGraph/weights_rank{rank}.txt", "a") as f:
        #    f.write(f"{weight.detach().cpu().tolist()}\n")                            
        
        ce = F.cross_entropy(input, target, reduction='none',
                             ignore_index=self.ignore_index)
        mask = target != self.ignore_index
        loss = torch.where(mask, weight[target.clamp(min=0)] * ce,
                           torch.zeros_like(ce))
        return loss.mean()
