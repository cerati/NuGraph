"""Transform to prune graph edges enforcing a maximum node degree."""
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected, contains_isolated_nodes, coalesce

from pynuml.data import NuGraphData


class PruneGraph(BaseTransform):
    """Prune planar graph edges to enforce a maximum node degree.

    For each node exceeding max_degree, retains a uniformly-spaced subset of
    neighbors sorted by wire-time distance, always keeping the nearest and
    farthest neighbors.

    Args:
        planes: List of detector plane names
        max_degree: Maximum allowed node degree after pruning
    """

    def __init__(self, planes: list[str], max_degree: int = 10):
        super().__init__()
        self.planes = planes
        self.max_degree = max_degree

    def forward(self, data: NuGraphData) -> NuGraphData:
        for p in self.planes:
            pos = data[p].pos[:, :2]  # wire, time coordinates
            edge_index = data[p, 'plane', p].edge_index
            data[p, 'plane', p].edge_index = self._prune(
                edge_index, pos, data[p].num_nodes)
        return data

    @staticmethod
    def _uniform_subset(n: int, max_degree: int) -> torch.Tensor:
        """Return exactly max_degree indices uniformly spaced in [0, n-1],
        including 0 (nearest) and n-1 (farthest). Deterministic, no randomness."""
        return torch.linspace(0, n - 1, max_degree).round().long()

    def _prune(self, edges: torch.Tensor, pos: torch.Tensor,
               num_nodes: int) -> torch.Tensor:
        # canonical directed form: row 0 <= row 1, no duplicate pairs
        directed = coalesce(torch.sort(edges, dim=0)[0])

        # node degree from undirected edges (both directions present in edges)
        degree = torch.zeros(num_nodes, dtype=torch.long)
        degree.scatter_add_(0, edges[0], torch.ones(edges.size(1), dtype=torch.long))

        # start with all edges retained; prune over-degree nodes one at a time
        keep_mask = torch.ones(directed.size(1), dtype=torch.bool)

        for node in (degree > self.max_degree).nonzero(as_tuple=True)[0]:
            # consider only edges still marked as kept (accounts for degree
            # reductions caused by processing earlier nodes in this loop)
            incident = ((directed[0] == node) | (directed[1] == node)) & keep_mask
            incident_idxs = incident.nonzero(as_tuple=True)[0]

            if incident_idxs.size(0) <= self.max_degree:
                continue

            # neighbor at the other end of each incident edge
            neighbors = torch.where(
                directed[0, incident_idxs] == node,
                directed[1, incident_idxs],
                directed[0, incident_idxs])

            # sort incident edges by wire-time distance to the neighbor
            order = torch.argsort(torch.norm(pos[neighbors] - pos[node], dim=-1))
            sorted_global = incident_idxs[order]

            # select uniformly-spaced subset keeping nearest and farthest
            local_keep = self._uniform_subset(sorted_global.size(0), self.max_degree)

            # drop all incident edges, then reinstate the selected subset
            keep_mask[incident_idxs] = False
            keep_mask[sorted_global[local_keep]] = True

        pruned = directed[:, keep_mask]

        # recompute degree from the pruned directed edges before reconnecting
        pruned_degree = torch.zeros(num_nodes, dtype=torch.long)
        pruned_degree.scatter_add_(0, pruned[0], torch.ones(pruned.size(1), dtype=torch.long))
        pruned_degree.scatter_add_(0, pruned[1], torch.ones(pruned.size(1), dtype=torch.long))

        pruned = self._reconnect_isolated(pruned, pos, pruned_degree, num_nodes, self.max_degree)
        return to_undirected(pruned, num_nodes=num_nodes)

    @staticmethod
    def _reconnect_isolated(edges: torch.Tensor, pos: torch.Tensor,
                            degree: torch.Tensor, num_nodes: int,
                            max_degree: int) -> torch.Tensor:
        """Connect each isolated node to its closest node with degree < max_degree.

        degree is updated in-place as edges are added so that the available
        candidate pool stays accurate across multiple isolated nodes.
        """
        if not contains_isolated_nodes(edges, num_nodes=num_nodes):
            return edges

        all_nodes = torch.arange(num_nodes)
        is_isolated = ~torch.isin(all_nodes, edges)
        isolated = all_nodes[is_isolated]

        # degree here reflects the state after pruning; update it as we add edges
        degree = degree.clone()
        new_edges = []

        for iso in isolated:
            # candidates: any non-isolated node still below the degree limit
            eligible = (~is_isolated) & (degree < max_degree)
            if not eligible.any():
                # no eligible candidate exists; connect to the closest node
                # regardless of degree to avoid leaving a disconnected node
                eligible = ~is_isolated

            dists = torch.norm(pos[eligible] - pos[iso], dim=-1)
            target = all_nodes[eligible][torch.argmin(dists)]

            new_edges.append(torch.sort(torch.stack([iso, target]))[0])
            degree[iso] += 1
            degree[target] += 1

        return torch.cat([edges] + [e.unsqueeze(1) for e in new_edges], dim=1)
