from torch import Tensor, tensor_split, cat, einsum, stack, empty, zeros
import torch.nn as nn

class ClassLinear(nn.Module):
    """
    NuGraph2 module for linear transformations grouped by class.

    Each semantic class has independent weights, so the hidden state remains
    class-conditional throughout the network. Weights are stored as a single
    batched parameter [num_classes, out_features, in_features] and applied with
    a single einsum kernel rather than a per-class Python loop.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_classes: Number of semantic classes
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        # weight shape [C, out, in] mirrors nn.Linear's transposed convention
        self.weight = nn.Parameter(empty(num_classes, out_features, in_features))
        self.bias   = nn.Parameter(zeros(num_classes, out_features))
        nn.init.kaiming_uniform_(self.weight.view(num_classes * out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        NuGraph2 class linear module forward pass

        Args:
            x: Feature tensor of shape [N, num_classes, in_features]

        Returns:
            Tensor of shape [N, num_classes, out_features]
        """
        return einsum('nci,coi->nco', x, self.weight) + self.bias

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Transparently remap old ModuleList checkpoint keys (net.0.weight, ...)
        # to the new batched parameter layout so existing checkpoints keep working.
        old_keys = [k for k in state_dict if k.startswith('net.') and '.weight' in k]
        if old_keys:
            weights = [state_dict.pop(f'net.{i}.weight') for i in range(self.num_classes)]
            biases  = [state_dict.pop(f'net.{i}.bias')   for i in range(self.num_classes)]
            state_dict['weight'] = stack(weights, dim=0)  # [C, out, in]
            state_dict['bias']   = stack(biases,  dim=0)  # [C, out]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
