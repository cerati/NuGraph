import torch
from torch_geometric.transforms import BaseTransform

class SpacePointGraph(BaseTransform):
    '''
    Add a 3D k-nearest-neighbour graph over spacepoint ("sp") nodes

    Each spacepoint is connected to its k nearest neighbours by Euclidean
    distance, giving every node a bounded, predictable in-degree regardless
    of local point density (dense, ambiguous 2D-to-3D reconstruction regions
    included) — unlike a radius cutoff, whose degree grows unboundedly with
    local density. The resulting graph is directed (guaranteed in-degree k,
    not necessarily mutual nearest-neighbours), standard practice for
    point-cloud GNNs. This should eventually be computed once at
    dataset-processing time in pynuml.process.HitGraphProducer, mirroring
    how the 2D per-plane Delaunay edges are baked in there, once raw
    simulation output is available again to rerun the MPI processing
    pipeline. For now only already-processed .gnn.h5 files are available, so
    this graph is (re)built here on every load.

    Args:
        k: Number of nearest neighbours to connect each spacepoint to
        chunk_size: Number of query rows to compare against all spacepoints
            at once. Crowded events can have tens of thousands of
            spacepoints, and computing all pairwise distances in one shot is
            quadratic in memory (e.g. ~9.3GB at n=50,000 for a single
            event) — chunking the query rows bounds peak memory to
            chunk_size * n instead, without changing the result: each row is
            still compared against every candidate, just in row batches.
    '''
    def __init__(self, k: int, chunk_size: int = 1024):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size

    def forward(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        pos = data['sp'].pos
        n = pos.size(0)

        # clamp k so events with very few spacepoints don't error out; if
        # there's nothing else to connect to (n <= 1), emit an empty graph
        k = min(self.k, n - 1)
        if k <= 0:
            data['sp', 'sp3d', 'sp'].edge_index = torch.empty((2, 0), dtype=torch.long)
            return data

        # equivalent to (but memory-bounded compared to):
        #   dist = torch.cdist(pos, pos)
        #   dist.fill_diagonal_(torch.inf)
        #   neighbours = dist.topk(k, largest=False).indices
        # each chunk still compares its rows against every one of the n
        # spacepoints, so results are identical to the full-matrix form above
        chunks = []
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            dist = torch.cdist(pos[start:end], pos)
            dist[torch.arange(end - start), torch.arange(start, end)] = torch.inf
            chunks.append(dist.topk(k, largest=False).indices)
        neighbours = torch.cat(chunks, dim=0) # [n, k]

        # for each node, k edges point in from its k nearest neighbours,
        # giving every node the exact same in-degree k (bar the n-1 clamp)
        dst = torch.arange(n).view(-1, 1).expand(-1, k).reshape(-1)
        src = neighbours.reshape(-1)

        data['sp', 'sp3d', 'sp'].edge_index = torch.stack((src, dst), dim=0)
        return data
