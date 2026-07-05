import torch
from torch_geometric.transforms import BaseTransform

class SpacePointGraph(BaseTransform):
    '''
    Add a 3D radius graph over spacepoint ("sp") nodes

    Two spacepoints are connected if their Euclidean distance is within
    `radius`. This should eventually be
    computed once at dataset-processing time in
    pynuml.process.HitGraphProducer, mirroring how the 2D per-plane Delaunay
    edges are baked in there, once raw simulation output is available again
    to rerun the MPI processing pipeline. For now only already-processed
    .gnn.h5 files are available in my case, so this graph is (re)built here on
    every load.

    Args:
        radius: Distance cutoff, in the same length units as data['sp'].pos (cm)
    '''
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    def __call__(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        pos = data['sp'].pos
        if pos.size(0) == 0:
            data['sp', 'sp3d', 'sp'].edge_index = torch.empty((2, 0), dtype=torch.long)
            return data

        # a spacepoint's distance to itself is always zero, so the diagonal
        # of this comparison is always true, giving every node a self-loop
        # for free, even if it has no neighbour within radius. the matrix is
        # symmetric, so the resulting edges are undirected by construction.
        adj = torch.cdist(pos, pos) <= self.radius
        src, dst = adj.nonzero(as_tuple=True)

        data['sp', 'sp3d', 'sp'].edge_index = torch.stack((src, dst), dim=0)
        return data
