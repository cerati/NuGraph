"""NuGraph3 encoder"""
import torch
from pynuml.data import NuGraphData
from ...util import InputNorm

class Encoder(torch.nn.Module):
    """
    NuGraph3 encoder

    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        sp_features: Number of real spacepoint input features (0 unless
            --3dfeatext is enabled, which provides delta_T, chi2, x, y, z)
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 instance_features: int,
                 sp_features: int = 0):
        super().__init__()

        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)

        # object condensation beta encoder
        self.beta_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1),
            torch.nn.Sigmoid(),
        )

        # object condensation coordinate encoder
        self.coord_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, instance_features),
            torch.nn.Mish(),
        )

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        # optional spacepoint input encoder (used when --3dfeatext is enabled)
        if sp_features > 0:
            self.sp_input_norm = InputNorm(sp_features)
            self.sp_net = torch.nn.Sequential(
                torch.nn.Linear(sp_features, nexus_features),
                torch.nn.Mish())
        else:
            self.sp_input_norm = None
            self.sp_net = None

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass

        Args:
            data: Graph data object
        """
        x_in = self.input_norm(data["hit"].x)
        data["hit"].x = self.planar_net(x_in)
        data["hit"].of = self.beta_net(x_in)
        data["hit"].ox = self.coord_net(x_in)

        if self.sp_net is not None:
            # encode real spacepoint input features as the initial sp embedding
            data["sp"].x = self.sp_net(self.sp_input_norm(data["sp"].x))
        else:
            data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                       self.nexus_features,
                                       device=data["hit"].x.device)

        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)