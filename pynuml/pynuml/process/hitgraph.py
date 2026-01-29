from typing import Any, Callable
import numpy as np
import pandas as pd

import torch
import torch_geometric as pyg

from .base import ProcessorBase
#changed: local to global
#changed: add spacepoint cuts
#changed: remove instance_label
#changed: revert to hits by plane
class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 label_position: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['global_wire','global_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 store_detailed_truth: bool = False):

        #print("__init__ is called")

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.label_position = label_position
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.store_detailed_truth = store_detailed_truth

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))

        super().__init__(file)

    @property
    def columns(self) -> dict[str, list[str]]:
        groups = {
            'hit_table': ['hit_id','global_plane','global_time','global_wire','integral','rms'],
            'spacepoint_table': []
        }
        if self.semantic_labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process','from_nu']
            groups['edep_table'] = []
        if self.event_labeller:
            groups['event_table'] = ['is_cc', 'nu_pdg']
        if self.label_vertex:
            keys = ['nu_vtx_corr','nu_vtx_wire_pos','nu_vtx_wire_time']
            if 'event_table' in groups:
                groups['event_table'].extend(keys)
            else:
                groups['event_table'] = keys
        if self.label_position:
            groups["edep_table"] = []
        return groups

    @property
    def metadata(self):
        print("metadata function in hitgraph.py is called")
        metadata = { 'planes': self.planes }
        if self.semantic_labeller is not None:
            metadata['semantic_classes'] = self.semantic_labeller.labels[:-1]
        else:
            print("self.semantic_labeller is apparently NONE")
        if self.event_labeller is not None:
            metadata['event_classes'] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:
        #print("__call__ is called (should pass through all events and thus 36814 entries (as in evt.h5)")
        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        hits = evt['hit_table']
        #spacepoints = evt['spacepoint_table'].reset_index(drop=True)
        spacepoints = evt['spacepoint_table'].query('Chi_squared<=0.5').reset_index(drop=True)#applies a chi2 filter
        # discard any events with less than 3 spacepoints
        if len(spacepoints)<3:
            print("skipping event because spacepoints are less than 3")
            return evt.name, None

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        #discard any events with a non-scalar/duplicate g4_id
        particle  = evt['particle_table']
        g4_duplicated_mask = particle.duplicated(subset='g4_id', keep=False)
        g4_duplicates = particle[g4_duplicated_mask]

        if not g4_duplicates.empty:
            print('skipping event with duplicate g4_id')
 #           r, sr, e, cryoID = evt.event_id
 #           print('----- EVENT -----')
 #           print('run ', r, "subrun ", sr)
 #           print('event ', e)
            grouped_ = g4_duplicates.groupby('g4_id')
 #           for key, group in grouped_:
 #               count = group['g4_id'].count()
 #               print(f"Group: {key} | g4_id count: {count}")
 #               print(group['g4_id'])
 #               print('-' * 40)
 #           print('----- END of previous EVENT -----')
            return evt.name, None

        # handle energy depositions
        if self.semantic_labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility

            # get ID of max particle
            g4_id = edeps[[energy_col, 'g4_id', 'hit_id']]
            g4_id = g4_id.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = g4_id.merge(hits, on='hit_id', how='right')

            # charge-weighted average of 3D position
            if self.label_position:
                edeps = edeps[["hit_id", "energy", "x_position", "y_position", "z_position"]]
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] *= edeps.energy
                edeps = edeps.groupby("hit_id").sum()
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] /= edeps.energy
                edeps = edeps.drop("energy", axis="columns")
                hits = edeps.merge(hits, on="hit_id", how="right")

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        for i in range(len(self.planes)):
            planehits = hits[hits.global_plane==i]
            nhits = planehits.filter_label.sum() if self.semantic_labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                print("skipping events with fewer than lower_bound simulated hits in any plane")
                return evt.name, None

        #r, sr, e = evt.event_id
        #if r!=7046 or sr!=112 or e!=5619: return evt.name, None
        #print('evt.event_id=',evt.event_id,' nu_pdg=',evt['event_table'].nu_pdg)

        # get labels for each particle
        if self.semantic_labeller:
            #print(self.semantic_labeller)
            #print('particle table:',evt['particle_table'])
            particles = self.semantic_labeller(evt['particle_table'])
            #print('particles=',particles)
            if particles is not None:
                try:
                    hits = hits.merge(particles, on='g4_id', how='left')
                    hits = hits.merge(evt['particle_table'][['from_nu','g4_id']], on='g4_id', how='left')

                   #consider cosmics as background -- obslete now: use y_filter and y_semantic
                   # hits.loc[hits['from_nu']==0, "semantic_label"] = -1
                except:
                    print('exception occurred when merging hits and particles')
                    print('hit table:', hits)
                    print('particle table:', particles)
                    print('skipping this event')
                    return evt.name, None
            else:
                print('empty particles')
                #return evt.name, None
                hits['semantic_label'] = np.nan
            #print('hit table:', hits)
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits out of {len(hits)} hits, skipping.')
#                print("G4_ID: ", hits.g4_id)
#                print("SEMANTIC LABEL: ", hits.semantic_label)
                return evt.name, None
            del mask

        data = pyg.data.HeteroData()

        # event metadata
        r, sr, e, cryoID, beamName = evt.event_id
        data['metadata'].run = r
        data['metadata'].subrun = sr
        data['metadata'].event = e
        data['metadata'].cryoID = cryoID
        data['metadata'].beamName = beamName
        #print("run ", r, "subrun", sr, "event", e, "cryo", cryoID)

        # spacepoint nodes
        if "position_x" in spacepoints.keys():
            if spacepoints.shape[0]>0:
                data["sp"].pos = torch.tensor(spacepoints[[f"position_{c}" for c in ("x", "y", "z")]].values).float()
            else:
                #data["sp"].pos = torch.empty(1, 0, 3) #would be correct, but does not work based on how torch_geometric/data/collate.py works
                data["sp"].pos = torch.tensor([[-999., -999., -999.]]).float()
        else:
            data['sp'].num_nodes = spacepoints.shape[0]

        # draw graph edges
        for i, plane_hits in hits.groupby('global_plane'):

            p = self.planes[i]
            plane_hits = plane_hits.reset_index(drop=True).reset_index(names='index_2d')

            # node position
            pos = torch.tensor(plane_hits[self.node_pos].values).float()
            data[p].pos = pos * self.pos_norm[None,:]

            # node features
            data[p].x = torch.tensor(plane_hits[self.node_feats].values).float()

            # node true position
            if self.label_position:
                data[p].c = torch.tensor(plane_hits[["x_position", "y_position", "z_position"]].values).float()

            # hit indices
            data[p].id = torch.tensor(plane_hits['hit_id'].values).long()

            # 2D edges
            data[p, 'plane', p].edge_index = self.transform(data[p]).edge_index

            # 3D edges
            edge3d = spacepoints.merge(plane_hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                       on=f'hit_id_{p}',
                                       how='inner')
            edge3d = edge3d[[f'index_2d_{p}','index_3d']].values.transpose()
            edge3d = torch.tensor(edge3d) if edge3d.size else torch.empty((2,0))
            data[p, 'nexus', 'sp'].edge_index = edge3d.long()

            # truth information
            f1_sem_nu = (plane_hits['from_nu'].fillna(0.0).astype(bool)) | (plane_hits['semantic_label'].fillna(-1) != -1)
            #keeps True from_nu=True OR sem_lab>=0 (for noise filter AND/OR semantic loss on nu only)

#            f1_filter_all = (plane_hits['from_nu'].fillna(0.0))
             #keeps True from_nu=True (noise+cosmics filter)

            sem_mask = plane_hits['semantic_label'].fillna(-1).astype(int).values
            from_nu_mask = plane_hits['from_nu'].fillna(0.0).astype(bool).values

#            sem_no_cosmics = np.where(from_nu_mask, sem_mask, -1) #keep sem value if from_nu otherwise sem = -1
            sem_w_cosmics = np.where(f1_sem_nu, sem_mask, -1) #keep sem value if filter in truth is 1

            data[p].y_filter = torch.tensor(f1_sem_nu.values).float()

            #adding from_nu to data
            from_nu_tensor = torch.tensor(from_nu_mask).bool()  # or .float() if you prefer
            data[p].from_nu = from_nu_tensor


            if self.semantic_labeller:
                data[p].y_semantic = torch.tensor(sem_w_cosmics).long() #exclude cosmics from semantic loss (ignore_index -1)
#                data[p].y_semantic = torch.tensor(plane_hits['semantic_label'].fillna(-1).values).long() #o.g. no excluding cosmics/from_nu
                data[p].y_semantic[data[p].y_semantic > 4] = -1
                #data[p].y_instance = torch.tensor(plane_hits['instance_label'].fillna(-1).values).long()
                if self.store_detailed_truth:
                    data[p].g4_id = torch.tensor(plane_hits['g4_id'].fillna(-1).values).long()
                    data[p].parent_id = torch.tensor(plane_hits['parent_id'].fillna(-1).values).long()
                    data[p].pdg = torch.tensor(plane_hits['type'].fillna(-1).values).long()
            if self.label_vertex:
                vtx_2d = torch.tensor([ event[f'nu_vtx_wire_pos_{i}'], event.nu_vtx_wire_time ]).float()
                data[p].y_vtx = vtx_2d * self.pos_norm[None,:]

        # event label
        if self.event_labeller:
            data['evt'].y = torch.tensor(self.event_labeller(event)).long()

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data
