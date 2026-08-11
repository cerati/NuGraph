import torch
from torch_geometric.transforms import BaseTransform
from torch import cat, stack

class NexusFeatures(BaseTransform):
  '''
    Add two features to the nexus nodes,
    chi-squared and maximum drift time spread,
    which encode the quality of the 3D SpacePoints
  '''
  def __init__(
    self,
    planes: list[str]
  ):
    super().__init__()
    self.planes = planes

  def forward(
    self,
    data: 'pyg.data.HeteroData'
    ) -> 'pyg.data.HeteroData':

    n_sp = data['sp'].num_nodes
    times, sigmas = [], []

    for p in self.planes:
      edge_index = data[p, 'nexus', 'sp'].edge_index
      hit_idx = edge_index[0]
      sp_idx  = edge_index[1]

      t = data[p].pos[hit_idx, 1]          # drift time for each hit
      s = data[p].x[hit_idx, 1].clamp(min=1e-6)  # rms for each hit
      w = 1.0 / s.square()                 # inverse-variance weights

      # sigma-weighted scatter to one value per spacepoint
      sum_wt = t.new_zeros(n_sp).index_add(0, sp_idx, w * t)
      sum_w  = t.new_zeros(n_sp).index_add(0, sp_idx, w)

      has_hit = sum_w > 0

      # spacepoints with hits: weighted mean time, combined sigma
      # spacepoints missing a hit from this plane: assign very large sigma
      # so their weight in the cross-plane chi2 is negligible
      t_sp = torch.where(has_hit, sum_wt / sum_w.clamp(min=1e-6), sum_wt)
      s_sp = torch.where(has_hit, 1.0 / sum_w.clamp(min=1e-6).sqrt(),
                         torch.full_like(sum_w, 1e6))

      times.append(t_sp)
      sigmas.append(s_sp)

    times  = stack(times, dim=1)   # [n_sp, n_planes]
    sigmas = stack(sigmas, dim=1)  # [n_sp, n_planes]

    # max drift time spread: only over planes that have a hit (large-sigma
    # planes carry t=0 which would distort the range, so mask them out)
    has_data = sigmas < 1e5
    t_max = times.masked_fill(~has_data, float('-inf')).amax(dim=1)
    t_min = times.masked_fill(~has_data, float( 'inf')).amin(dim=1)
    delta_T = (t_max - t_min).clamp(min=0.).unsqueeze(1)

    # chi-squared: weighted average of time across planes, weighted by 1/sigma^2
    # missing planes have w≈0 and contribute negligibly
    w2 = 1.0 / sigmas.square()
    t_weigh_avg = (w2 * times).sum(dim=1, keepdim=True) / w2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    s_weigh_avg = 1.0 / w2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    chi2 = (w2 * (times - t_weigh_avg).square()).sum(dim=1).unsqueeze(1)

    # nexus nodes get [delta_T, chi2, x, y, z] features: the two quality
    # features above, plus the real 3D position, which is otherwise never
    # fed into the network as a feature (only used for graph construction)
    data['sp'].x = cat((delta_T, chi2, data['sp'].pos), dim=1)

    return data