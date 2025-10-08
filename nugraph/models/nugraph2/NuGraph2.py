import argparse
import warnings
import psutil

import torch
from torch import Tensor, cat, empty, norm, topk, stack, log, zeros
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch

from .encoder import Encoder
from .plane import PlaneNet
from .nexus import NexusNet
from .decoders import SemanticDecoder, FilterDecoder

from ...data import H5DataModule

class NuGraph2(LightningModule):
    """PyTorch Lightning module for model training.

    Wrap the base model in a LightningModule wrapper to handle training and
    inference, and compute training metrics."""
    def __init__(self,
                 in_features: int = 4,
                 planar_features: int = 64,
                 nexus_features: int = 16,
                 planes: list[str] = ['u','v','y'],
                 semantic_classes: list[str] = ['MIP','HIP','shower','michel','diffuse'],
                 num_iters: int = 5,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 checkpoint: bool = False,
                 lr: float = 0.001):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.num_iters = num_iters
        self.lr = lr

        self.encoder = Encoder(in_features,
                               planar_features,
                               planes,
                               semantic_classes)

        self.plane_net = PlaneNet(in_features,
                                  planar_features,
                                  len(semantic_classes),
                                  planes,
                                  checkpoint=checkpoint)

        self.nexus_net = NexusNet(planar_features,
                                  nexus_features,
                                  len(semantic_classes),
                                  planes,
                                  checkpoint=checkpoint)

        self.semantic_decoder = SemanticDecoder(planar_features, planes, semantic_classes)
        self.filter_decoder = FilterDecoder(planar_features, planes, semantic_classes)
        self.decoders: List[Union[SemanticDecoder,FilterDecoder]] = [self.semantic_decoder,self.filter_decoder]

        #self.decoders = []
        #
        #if semantic_head:
        #    self.semantic_decoder = SemanticDecoder(
        #        planar_features,
        #        planes,
        #        semantic_classes)
        #    self.decoders.append(self.semantic_decoder)
        #
        #if filter_head:
        #    self.filter_decoder = FilterDecoder(
        #        planar_features,
        #        planes,
        #        semantic_classes)
        #    self.decoders.append(self.filter_decoder)

        if len(self.decoders) == 0:
            raise Exception('At least one decoder head must be enabled!')

    def forward(self,
                x: dict[str, Tensor],
                edge_index_plane: dict[str, Tensor],
                edge_index_nexus: dict[str, Tensor],
                nexus: Tensor,
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        #print(x)
        #print(edge_index_plane)
        # need to undo norm
        # sample: /exp/icarus/data/users/cerati/NuGraph/icarus-mpvmpr-new-00-04.gnn.h5
        #normf = {'u': torch.tensor([[166.21169  , 178.42711  , 276.38657  ,   3.9433618], [ 77.15124  ,  64.01552  , 290.06683  ,   1.3364509]], dtype=torch.float32),
        #         'v': torch.tensor([[1264.9119   ,  177.53986  ,  330.69482  ,    4.4075446], [ 249.62796  ,   63.493645 ,  323.87247  ,    1.473314 ]], dtype=torch.float32),
        #         'y': torch.tensor([[1256.1769   ,  180.19048  ,  321.59045  ,    4.3928657], [ 257.89337  ,   63.59527  ,  316.3936   ,    1.4808525]], dtype=torch.float32)}
        # sample: /exp/icarus/data/users/shseo/NuGraphH5-BNB-merge/mpvmpr_bnb_numu_cos.gnn.h5
        #normf = {'u': torch.tensor([[168.04594  , 178.3245   , 266.6149   ,   3.857218 ], [ 82.80644  ,  67.60649  , 274.32666  ,   1.2912455]], dtype=torch.float32),
        #         'v': torch.tensor([[1245.3547   ,  176.54117  ,  323.52786  ,    4.3267984], [ 293.06314  ,   66.8194   ,  322.11386  ,    1.4249923]], dtype=torch.float32),
        #         'y': torch.tensor([[1225.5012   ,  183.58075  ,  310.83493  ,    4.3409133], [ 307.1943   ,   67.063324 ,  312.461    ,    1.4532351]], dtype=torch.float32)}
        # sample: /exp/uboone/data/users/cerati/exatrkx-train-mcc10/numi_all.gnn.h5
        #normf = {'u': torch.tensor([[394.7723   , 181.16737  , 159.11298  ,   4.6349325], [148.07278  ,  76.55051  , 346.6077   ,   2.2700603]], dtype=torch.float32),
        #         'v': torch.tensor([[374.3612   , 180.96751  , 153.33102  ,   4.457545 ], [148.27586  ,  78.48496  , 289.37195  ,   1.9274824]], dtype=torch.float32),
        #         'y': torch.tensor([[554.61926  , 182.04688  , 125.07064  ,   4.221719 ], [285.84528  ,  72.809746 , 144.16167  ,   1.5962849]], dtype=torch.float32)}
        # sample: /exp/uboone/data/users/cerati/exatrkx-train-mcc10/gnnfiles-copy/numi_all_withbkg.gnn.h5
        #normf = {'u': torch.tensor([[386.67844 , 178.79637 , 166.40633 ,   4.596251], [154.84381 ,  77.761345, 445.8406  ,   2.295901]], dtype=torch.float32),
        #         'v': torch.tensor([[369.44028 , 178.59866 , 152.50826 ,   4.388774], [152.19939 ,  79.518326, 343.1043  ,   1.932463]], dtype=torch.float32),
        #         'y': torch.tensor([[539.4748  , 179.91644 , 127.31915 ,   4.217183], [296.09464 ,  73.54983 , 148.6749  ,  1.6577382]], dtype=torch.float32)}
        # sample: /nugraph/numiallwr2.gnn.h5
        #normf = {'u': torch.tensor([[395.23712  , 180.31087  , 156.4287   ,   4.6503887], [146.59378  ,  76.942184 , 288.28412  ,   2.277651 ]], dtype=torch.float32),
        #         'v': torch.tensor([[374.18634  , 180.33629  , 152.55469  ,   4.465103 ], [147.33215  ,  78.70177  , 253.89346  ,   1.9274441]], dtype=torch.float32),
        #         'y': torch.tensor([[552.84753  , 181.09207  , 125.493675 ,   4.223127 ], [283.6226   ,  73.07375  , 159.50517  ,   1.5871835]], dtype=torch.float32)}
        # feature extension
        #for p in self.planes:
        #    #break #no extended features
        #    #print(edge_index_plane[p])
        #    if len(edge_index_plane[p])<2 or edge_index_plane[p].shape[1]<3:
        #        dwire = zeros(x[p].shape[0],1)
        #        dtime = zeros(x[p].shape[0],1)
        #        nodes_degree = zeros(x[p].shape[0],1)
        #    else:
        #        #print(p)
        #        #print(x[p].size())
        #        # Adding delta wire an delta time (dwire/dtime doesn't work; some infs)
        #        # Extracting wire and time information
        #        wt_coords = stack((x[p][:, 0]*normf[p][1][0]+normf[p][0][0], x[p][:, 1]*normf[p][1][1]+normf[p][0][1]), dim=1) # [wire, time], after I normalized them
        #        # Calculating pairwise euclidean distances of nodes in the wire vs time space
        #        #print('wt_coords',wt_coords.size())
        #        #print(wt_coords[:, None, :].size())
        #        #print(wt_coords[None, :, :].size())
        #        #print('diff=',wt_coords[:, None, :] - wt_coords[None, :, :])
        #        dist_table = norm(wt_coords[:, None, :] - wt_coords[None, :, :], dim=-1)
        #        #print(dist_table)
        #        dist_table.fill_diagonal_(float('inf'))
        #        # Find a (n_nodes, 2) matrix containing the distances and indexes of the two closest nodes to each node
        #        dists_2closest_nodes, idxs_2closest_nodes = topk(dist_table, 2, dim=1, largest=False, sorted=True)
        #        # Finding the ratio of the wire and time differences of the two closest neighbors
        #        # Double delta (Giuseppe suggestion)
        #        dwire = (2*wt_coords[:, 0] - wt_coords[idxs_2closest_nodes[:,1], 0] - wt_coords[idxs_2closest_nodes[:,0], 0]).view(-1,1)
        #        dtime = (2*wt_coords[:, 1] - wt_coords[idxs_2closest_nodes[:,1], 1] - wt_coords[idxs_2closest_nodes[:,0], 1]).view(-1,1)
        #        ## Adding shortest edge length
        #        #min_dist = dists_2closest_nodes[:,0].view(-1,1) # 'dists_2closest_nodes' is sorted in ascending order
        #        ## Adding node degree
        #        nodes_degree = torch.unique(edge_index_plane[p][0], sorted=True, return_counts=True)[1].view(-1,1)
        #        nodes_degree = log(nodes_degree) # Should I use log(nodes_degree) instead?
        #
        #    # Extending the original node feature matrix with the new features
        #    x[p] = cat((x[p], dwire, dtime, nodes_degree), dim=-1)
        #    #x[p] = cat((x[p], dwire, dtime, nodes_degree, min_dist), dim=-1)

        # drop RMS, i.e. column index 3
        #for p in self.planes:
        #    column_index_to_drop = 3
        #    x[p] = torch.cat((x[p][:, :column_index_to_drop], x[p][:, column_index_to_drop+1:]), dim=1)
        #print(x)
        m = self.encoder(x)
        #print(m)
        for _ in range(self.num_iters):
            # shortcut connect features
            for i, p in enumerate(self.planes):
                s = x[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
                m[p] = torch.cat((m[p], s), dim=-1)
            self.plane_net(m, edge_index_plane)
            self.nexus_net(m, edge_index_nexus, nexus)
        #ret = {}
        #for decoder in self.decoders:
        #    ret.update(decoder(m, batch))
        ret: dict[str, dict[str, Tensor]] = {}
        ret.update(self.semantic_decoder(m, batch))
        ret.update(self.filter_decoder(m, batch))
        return ret

    def step(self, data: HeteroData | Batch):

        # if it's a single data instance, convert to batch manually
        if isinstance(data, Batch):
            batch = data
        else:
            batch = Batch.from_data_list([data])

        # unpack tensors to pass into forward function
        x = self(batch.collect('x'),
                 { p: batch[p, 'plane', p].edge_index for p in self.planes },
                 { p: batch[p, 'nexus', 'sp'].edge_index for p in self.planes },
                 torch.empty(batch['sp'].num_nodes, 0),
                 { p: batch[p].batch for p in self.planes })

        # append output tensors back onto input data object
        if isinstance(data, Batch):
            dlist = [ HeteroData() for i in range(data.num_graphs) ]
            for attr, planes in x.items():
                for p, t in planes.items():
                    if t.size(0) == data[p].num_nodes:
                        tlist = unbatch(t, data[p].batch)
                    elif t.size(0) == data.num_graphs:
                        tlist = unbatch(t, torch.arange(data.num_graphs))
                    else:
                        raise Exception(f'don\'t know how to unbatch attribute {attr}')
                    for it_d, it_t in zip(dlist, tlist):
                        it_d[p][attr] = it_t
            tmp = Batch.from_data_list(dlist)
            data.update(tmp)
            for attr, planes in x.items():
                for p in planes:
                    data._slice_dict[p][attr] = tmp._slice_dict[p][attr]
                    data._inc_dict[p][attr] = tmp._inc_dict[p][attr]

        else:
            for key, value in x.items():
                data.set_value_dict(key, value)

        self.data = data

    def on_train_start(self):
        hpmetrics = { 'max_lr': self.hparams.lr }
        self.logger.log_hyperparams(self.hparams, metrics=hpmetrics)
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

        scalars = {
            'loss': {'loss': [ 'Multiline', [ 'loss/train', 'loss/val' ]]},
            'acc': {}
        }
        for c in self.semantic_classes:
            scalars['acc'][c] = [ 'Multiline', [
                f'semantic_accuracy_class_train/{c}',
                f'semantic_accuracy_class_val/{c}'
            ]]
        self.logger.experiment.add_custom_scalars(scalars)

    def training_step(self,
                      batch,
                      batch_idx: int) -> float:
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'train')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/train', total_loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log_memory(batch, 'train')
        return total_loss

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'val', True)
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/val', total_loss, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'test', True)
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/test', total_loss, batch_size=batch.num_graphs)
        self.log_memory(batch, 'test')

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Batch,
                     batch_idx: int = 0) -> Batch:
        self.step(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        onecycle = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    def log_memory(self, batch: Batch, stage: str) -> None:
        # log CPU memory
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.
        cpu_mem = psutil.Process().memory_info().rss / float(1073741824)
        self.max_mem_cpu = max(self.max_mem_cpu, cpu_mem)
        self.log(f'memory_cpu/{stage}', self.max_mem_cpu,
                 batch_size=batch.num_graphs, reduce_fx=torch.max)

        # log GPU memory
        if not hasattr(self, 'max_mem_gpu'):
            self.max_mem_gpu = 0.
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / float(1073741824)
            self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batch.num_graphs, reduce_fx=torch.max)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        '''Add argparse argpuments for model structure'''
        model = parser.add_argument_group('model', 'NuGraph2 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=4,
                           help='Number of input node features')
        model.add_argument('--planar-feats', type=int, default=64,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=16,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--semantic', action='store_true', default=False,
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true', default=False,
                           help='Enable background filter head')
        model.add_argument('--no-checkpointing', action='store_true', default=False,
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph2':
        return cls(
            in_features=args.in_feats,
            planar_features=args.planar_feats,
            nexus_features=args.nexus_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            num_iters=args.num_iters,
            semantic_head=args.semantic,
            filter_head=args.filter,
            checkpoint=not args.no_checkpointing,
            lr=args.learning_rate)
