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
        instance_features: Number of instance node features
        input_edge_geom: If True, compute 5 fixed geometric features on hit-hit edges
        input_nexus_feats: If True, compute spacepoint quality features (delta_T, chi2, x, y, z)
            from contributing hits and encode them as the initial sp embedding
        instance_ox_residual: If True, stash the initial (pre-message-passing) condensation
            coordinates as data["hit"].ox_initial, for the instance decoder's residual shortcut
        instance_nexus_agg_initial: If True, stash the initial (pre-message-passing) sp
            embedding as data["sp"].x_initial, for the instance decoder's nexus-aggregation
            shortcut. Only meaningful when input_nexus_feats is also True — otherwise the
            stashed value is just zeros.
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 instance_features: int,
                 input_edge_geom: bool = False,
                 input_nexus_feats: bool = False,
                 instance_ox_residual: bool = False,
                 instance_nexus_agg_initial: bool = False):
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
        self.input_edge_geom = input_edge_geom
        self.input_nexus_feats = input_nexus_feats
        self.instance_ox_residual = instance_ox_residual
        self.instance_nexus_agg_initial = instance_nexus_agg_initial

        # optional spacepoint feature encoder: normalises and encodes
        # [delta_T, chi2, x, y, z] computed on-the-fly in forward()
        if input_nexus_feats:
            self.sp_input_norm = InputNorm(5)
            self.sp_net = torch.nn.Sequential(
                torch.nn.Linear(5, nexus_features),
                torch.nn.Mish())

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass

        Args:
            data: Graph data object
        """
        x_in = self.input_norm(data["hit"].x)

        if self.input_edge_geom:
            # Compute fixed geometric edge features from normalized hit inputs.
            # Recomputed each forward pass since hit features differ per batch.
            pp = data["hit", "delaunay-planar", "hit"]
            src, dst = pp.edge_index
            d_wire     = x_in[src, 0] - x_in[dst, 0]
            d_time     = x_in[src, 1] - x_in[dst, 1]
            d_integral = x_in[src, 2] - x_in[dst, 2]
            d_rms      = x_in[src, 3] - x_in[dst, 3]
            distance   = torch.hypot(d_wire, d_time)
            pp.edge_geom = torch.stack([d_integral, d_rms, d_wire, d_time, distance], dim=1)

        if self.input_nexus_feats:
            # Compute spacepoint quality features from raw hit inputs via nexus edges.
            # Must happen before data["hit"].x is overwritten by planar_net so that
            # columns 1 (drift time) and 3 (rms) still refer to the physical inputs.
            # Reading raw (non-differentiable) data["hit"].x here also avoids an
            # autograd version conflict: hit_idx would be saved for backward if we
            # indexed into a differentiable tensor, and PyG's propagate can later
            # touch edge_index in-place, incrementing hit_idx's version.
            edge_index = data["hit", "nexus", "sp"].edge_index
            hit_idx, sp_idx = edge_index[0], edge_index[1]
            n_sp = data["sp"].num_nodes

            t_all = data["hit"].x[hit_idx, 1]   # raw drift time (col 1 after NG3 Transform)
            s_all = data["hit"].x[hit_idx, 3]   # raw rms     (col 3 after NG3 Transform)
            plane_all = data["hit"].plane[hit_idx]
            n_planes = int(data["hit"].plane.max().item()) + 1

            times, sigmas = [], []
            for p in range(n_planes):
                mask = plane_all == p
                t_p = t_all[mask]
                s_p = s_all[mask].clamp(min=1e-6)
                sp_p = sp_idx[mask]
                w = 1.0 / s_p.square()
                sum_wt = t_all.new_zeros(n_sp).index_add(0, sp_p, w * t_p)
                sum_w  = t_all.new_zeros(n_sp).index_add(0, sp_p, w)
                has_hit = sum_w > 0
                t_sp = torch.where(has_hit, sum_wt / sum_w.clamp(min=1e-6), sum_wt)
                s_sp = torch.where(has_hit, 1.0 / sum_w.clamp(min=1e-6).sqrt(),
                                   torch.full_like(sum_w, 1e6))
                times.append(t_sp)
                sigmas.append(s_sp)

            times  = torch.stack(times, dim=1)   # [n_sp, n_planes]
            sigmas = torch.stack(sigmas, dim=1)

            has_data = sigmas < 1e5
            t_max = times.masked_fill(~has_data, float('-inf')).amax(dim=1)
            t_min = times.masked_fill(~has_data, float( 'inf')).amin(dim=1)
            delta_T = (t_max - t_min).clamp(min=0.).unsqueeze(1)

            w2 = 1.0 / sigmas.square()
            t_avg = ((w2 * times).sum(dim=1, keepdim=True)
                     / w2.sum(dim=1, keepdim=True).clamp(min=1e-6))
            chi2 = (w2 * (times - t_avg).square()).sum(dim=1).unsqueeze(1)

            sp_feats = torch.cat([delta_T, chi2, data["sp"].pos], dim=1)
            data["sp"].x = self.sp_net(self.sp_input_norm(sp_feats))
            if self.instance_nexus_agg_initial:
                data["sp"].x_initial = data["sp"].x
        else:
            data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                       self.nexus_features,
                                       device=data["hit"].x.device)

        data["hit"].x = self.planar_net(x_in)
        data["hit"].of = self.beta_net(x_in)
        data["hit"].ox = self.coord_net(x_in)
        if self.instance_ox_residual:
            data["hit"].ox_initial = data["hit"].ox

        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
