from argparse import ArgumentParser
import warnings

import h5py
import tqdm

from torch import tensor, cat
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning import LightningDataModule

from ..data import H5Dataset
from ..util import PositionFeatures, FeatureNormMetric, FeatureNorm

class H5DataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 prepare: bool = False):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        self.filename = data_path
        self.batch_size = batch_size

        with h5py.File(self.filename) as f:

            # load metadata
            try:
                self.planes = f['planes'].asstr()[()].tolist()
                self.semantic_classes = f['semantic_classes'].asstr()[()].tolist()
                self.event_classes = f['event_classes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes", "semantic_classes" and "event classes" are required.')
                exit()

            # load sample splits
            try:
                train_samples = f['samples/train'].asstr()[()]
                val_samples = f['samples/validation'].asstr()[()]
                test_samples = f['samples/validation'].asstr()[()]
            except:
                print('Sample splits not found in file! Call "generate_samples" to create them.')
                exit()

            # load feature normalisations
            try:
                norm = {}
                for p in self.planes:
                    norm[p] = tensor(f[f'norm/{p}'][()])
            except:
                print('Feature normalisations not found in file! Call "generate_norm" to create them.')
                exit()

        transform = Compose((PositionFeatures(self.planes),
                             FeatureNorm(self.planes, norm)))

        self.train_dataset = H5Dataset(self.filename, train_samples, transform)
        self.val_dataset = H5Dataset(self.filename, val_samples, transform)
        self.test_dataset = H5Dataset(self.filename, test_samples, transform)

    @staticmethod
    def generate_samples(data_path: str):
        with h5py.File(data_path, 'r+') as f:
            samples = list(f['dataset'].keys())
            split = int(0.05 * len(samples))
            splits = [ len(samples)-(2*split), split, split ]
            train, val, test = random_split(samples, splits)

            for key in [ 'train', 'validation', 'test' ]:
                name = f'samples/{key}'
                if f.get(f'samples/{key}') is not None:
                    del f[f'samples/{key}']

            f['samples/train'] = list(train)
            f['samples/validation'] = list(val)
            f['samples/test'] = list(test)

    @staticmethod
    def generate_norm(data_path: str, batch_size: int):
        with h5py.File(data_path, 'r+') as f:
            # load plane metadata
            try:
                planes = f['planes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" is required.')
                exit()

            loader = DataLoader(H5Dataset(data_path,
                                          list(f['dataset'].keys()),
                                          PositionFeatures(planes)),
                                batch_size=batch_size)

            print('  generating feature norm...')
            metrics = None
            for batch in tqdm.tqdm(loader):
                for p in planes:
                    if not metrics:
                        num_feats = batch[p].x.shape[-1]
                        metrics = { p: FeatureNormMetric(num_feats) for p in planes }
                    metrics[p].update(batch[p].x)
            for p in planes:
                f[f'norm/{p}'] = metrics[p].compute()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)

    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        data = parser.add_argument_group('data', 'Data module configuration')
        data.add_argument('--data-path', type=str,
                          default='/data/CHEP2023/filtered.gnn.h5',
                          help='Location of input data file')
        data.add_argument('--batch-size', type=int, default=64,
                          help='Size of each batch of graphs')
        data.add_argument('--limit_train_batches', type=int, default=None,
                          help='Max number of training batches to be used')
        data.add_argument('--limit_val_batches', type=int, default=None,
                          help='Max number of validation batches to be used')
        return parser