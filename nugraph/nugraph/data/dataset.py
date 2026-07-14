"""NuGraph dataset"""
from typing import Callable, Optional
import h5py
from torch_geometric.data import Dataset

from pynuml.data import NuGraphData

class NuGraphDataset(Dataset):
    """NuGraph dataset

    Args:
        filename: Name of dataset file
        samples: List of graph object dataset names in file
        transform: Transforms to apply to graph objects
    """
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.filename = filename
        self.samples = samples
        # opened lazily, once per worker process (see get()) — h5py file
        # handles can't be pickled, so eagerly opening here would break
        # multiprocessing DataLoader workers, which need to pickle this
        # dataset object to hand it to each worker
        self.file = None

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        if self.file is None:
            self.file = h5py.File(self.filename)
        key = f"/dataset/{self.samples[idx]}"
        return NuGraphData.load(self.file[key])
