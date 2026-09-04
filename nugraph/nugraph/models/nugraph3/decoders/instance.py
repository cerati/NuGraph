"""NuGraph3 instance decoder"""
from typing import Any
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import cumsum, unbatch
from ....util import ObjConLoss, RecallLoss
from ..types import Data, N_IT, E_H_IT, N_IP, E_H_IP

def _nexus_mean_agg(sp_feat: torch.Tensor, hit_idx: torch.Tensor,
                    sp_idx: torch.Tensor, num_hits: int) -> torch.Tensor:
    """Mean-aggregate a per-sp feature tensor onto hit nodes via nexus edges"""
    agg = sp_feat.new_zeros(num_hits, sp_feat.size(1)).index_add_(0, hit_idx, sp_feat[sp_idx])
    counts = sp_feat.new_zeros(num_hits).index_add_(
        0, hit_idx, sp_feat.new_ones(sp_idx.size(0))).clamp(min=1).unsqueeze(1)
    return agg / counts

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
        nexus_features: Number of nexus (sp) node features, required if nexus_agg or
            nexus_agg_initial is True
        ox_residual: If True, add a residual connection from the encoder's initial
            (pre-message-passing) condensation coordinates (data["hit"].ox_initial)
        nexus_agg: If True, mean-aggregate the final (post-message-passing) sp
            embedding onto hit nodes via nexus edges, as an additional residual branch
        nexus_agg_initial: If True, mean-aggregate the initial (pre-message-passing) sp
            embedding (data["sp"].x_initial) onto hit nodes via nexus edges, as an
            additional residual branch. May be combined with nexus_agg.
    """
    def __init__(self, hit_features: int, instance_features: int,
                 particle_loss: bool = False,
                 nexus_features: int = None,
                 ox_residual: bool = False,
                 nexus_agg: bool = False,
                 nexus_agg_initial: bool = False):
        super().__init__()

        # loss function
        self.loss = ObjConLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # beta MLP
        self.beta_net = nn.Sequential(
            nn.Linear(hit_features + 1, hit_features),
            nn.Mish(),
            nn.Linear(hit_features, 1),
            nn.Sigmoid(),
        )

        # coordinate MLP
        self.coord_net = nn.Sequential(
            nn.Linear(hit_features + instance_features, hit_features),
            nn.Mish(),
            nn.Linear(hit_features, instance_features),
        )

        self.ox_residual = ox_residual
        self.nexus_agg = nexus_agg
        self.nexus_agg_initial = nexus_agg_initial

        # optional nexus-aggregation residual branch: reads sp.x and/or sp.x_initial
        # via nexus edges, as a dedicated path that bypasses the shared h.x pathway
        n_sources = int(nexus_agg) + int(nexus_agg_initial)
        if n_sources > 0:
            self.nexus_coord_net = nn.Sequential(
                nn.Linear(hit_features + instance_features + n_sources * nexus_features,
                         hit_features),
                nn.Mish(),
                nn.Linear(hit_features, instance_features),
            )
        else:
            self.nexus_coord_net = None

        self.dbscan = DBSCAN(eps=0.3, min_samples=15)
        self.particle_loss = particle_loss

    # pylint: disable=arguments-differ
    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        h = data["hit"]
        device = h.x.device

        # run network and add output to graph object
        h.of = self.beta_net(torch.cat((h.x, h.of), dim=1)).squeeze(dim=-1)

        ox_in = h.ox
        if self.ox_residual:
            ox_in = ox_in + h.ox_initial
            del h.ox_initial  # transient, created post-batching with no slice_dict entry;
                               # must not survive into to_data_list() below

        h.ox = self.coord_net(torch.cat((h.x, ox_in), dim=1))

        if self.nexus_agg or self.nexus_agg_initial:
            sp = data["sp"]
            hit_idx, sp_idx = data["hit", "nexus", "sp"].edge_index
            aggs = []
            if self.nexus_agg:
                aggs.append(_nexus_mean_agg(sp.x, hit_idx, sp_idx, h.x.size(0)))
            if self.nexus_agg_initial:
                aggs.append(_nexus_mean_agg(sp.x_initial, hit_idx, sp_idx, h.x.size(0)))
                del sp.x_initial  # same transient-attribute concern as ox_initial above
            h_nexus_agg = torch.cat(aggs, dim=1)
            h.ox = h.ox + self.nexus_coord_net(torch.cat((h.x, ox_in, h_nexus_agg), dim=1))

        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = h.ptr
            data._slice_dict["hit"]["ox"] = h.ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate semantic loss to input to object condensation particle loss
        loss_semantic = None
        if (self.particle_loss):
            loss_semantic = data.hit_loss()

        # calculate loss
        loss = self.loss(h.ox, h.of, data.y_i(), h.y_semantic,
                         data[N_IT].num_nodes, data[E_H_IT].edge_index,
                         loss_semantic)
        loss *= (-1 * self.temp).exp()
        b, v, p = loss
        loss = loss.sum() + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v
            if self.particle_loss:
                metrics[f"instance/particle-loss-{stage}"] = p

        if not self.training:
            # add materialized instances
            mask = torch.ones_like(h.of, dtype=torch.bool)
            if hasattr(h, "x_filter"):
                mask = mask & (h.x_filter > 0.5)
            if hasattr(h, "x_semantic"):
                mask = mask & (h.x_semantic.argmax(dim=1) != 6)
            if isinstance(data, Batch):
                x_ip, e_h_ip = [], []
                for ox, m in zip(unbatch(h.ox, h.batch), unbatch(mask, h.batch)):
                    x, e = self.materialize(ox, m)
                    x_ip.append(x)
                    e_h_ip.append(e)

                # particle nodes
                data[N_IP].x = torch.cat(x_ip, dim=0)
                data[N_IP].batch = torch.cat(
                    [torch.full((0,), i, dtype=torch.long, device=device) for i, x in enumerate(x_ip)])
                data[N_IP].ptr = cumsum(torch.tensor([x.size(0) for x in x_ip], device=device))
                data._slice_dict[N_IP] = {"x": data[N_IP].ptr} # pylint: disable=protected-access
                data._inc_dict[N_IP] = { # pylint: disable=protected-access
                    "x": torch.zeros(data.num_graphs, dtype=torch.long, device=device)
                }

                # particle edges
                e_inc = torch.stack((h.ptr[:-1], data[N_IP].ptr[:-1]), dim=1).unsqueeze(2)
                data[E_H_IP].edge_index = torch.cat([e + inc for e, inc in zip(e_h_ip, e_inc)], dim=1)
                data._slice_dict[E_H_IP] = { # pylint: disable=protected-access
                    "edge_index": cumsum(torch.tensor([e.size(1) for e in e_h_ip]))
                }
                data._inc_dict[E_H_IP] = {"edge_index": e_inc} # pylint: disable=protected-access

                # calculate rand score per graph
                rand = []
                for l in data.to_data_list():
                    mask = l["hit"].y_semantic >= 0
                    rand.append(adjusted_rand_score(l.x_i()[mask], l.y_i()[mask]))
                rand = torch.stack(rand).mean()

            else:
                data[N_IP].x, data[E_H_IP].edge_index = self.materialize(h.ox, mask)
                rand = adjusted_rand_score(data.x_i(), data.y_i())

            if not -1. <= rand <= 1.:
                raise RuntimeError(f"Adjusted Rand Score metric value {rand} is outside allowed range!")

            if stage:
                metrics[f"instance/adjusted-rand-{stage}"] = rand

        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def materialize(self, ox: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize instance embedding
        
        Args:
            ox: object condensation embedding tensor
            mask: bool mask tensor for background hit removal
        """

        # if there are no signal hits to cluster, skip dbscan and return empty tensors
        if not mask.sum():
            x_ip = torch.empty(0, 0, dtype=ox.dtype, device=ox.device)
            e_h_ip = torch.empty(2, 0, dtype=torch.long, device=ox.device)
            return x_ip, e_h_ip

        i = torch.empty(ox.size(0), dtype=torch.long, device=ox.device).fill_(-1)
        arr = ox[mask].detach().to(torch.float32).cpu().numpy()
        labels = self.dbscan.fit_predict(arr)
        i[mask] = torch.from_numpy(labels).to(device=ox.device, dtype=torch.long)
        x_ip = torch.empty(i.max()+1, 0, dtype=ox.dtype, device=ox.device)
        mask = i > -1
        e_h_ip = torch.stack((torch.nonzero(mask).squeeze(1), i[mask])).long()
        return x_ip, e_h_ip

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
