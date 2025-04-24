from torch import cat
from torch_geometric.transforms import BaseTransform

class DropRMS(BaseTransform):
    '''Remove RMS from the node feature tensor'''
    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        for p in self.planes:
            data[p].x = data[p].x[:,:-1]
        return data
