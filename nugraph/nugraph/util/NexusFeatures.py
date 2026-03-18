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

  def __call__(
    self, 
    data: 'pyg.data.HeteroData'
    ) -> 'pyg.data.HeteroData':

    # get times and rms of hits contributing to each nexus node
    # this gets us a list of three [n_sp] tensors
    times, sigmas = [], []

    for p in self.planes:
      edge_index = data[p, 'nexus', 'sp'].edge_index 
      hit_idx = edge_index[0]
      times.append(data[p].pos[hit_idx, 1]) # wire, time; get time
      sigmas.append(data[p].x[hit_idx, 1])  # integral, rms; get rms

    # stack the tensors from [n_sp, 3]
    # stack wants tensors of the same shape and creates a new dimension
    times  = stack(times, dim=1)
    sigmas = stack(sigmas, dim=1)

    # max drift time spread: get the maximum/minimum time
    # for each sp, then take the difference and convert to [n_sp, 1]
    delta_T = (times.max(dim=1).values - times.min(dim=1).values).unsqueeze(1) 

    # chi-squared
    # weighted average of time, w.r.t. rms, [n_sp, 1]
    t_weigh_avg = ((t / (sigmas**2)).sum(dim=1) / (1 / (sigmas**2)).sum(dim=1)).unsqueeze(1)
    # rms of the weighted mean, [n_sp, 1]
    s_weigh_avg = (1 / (1 / sigmas**2).sum(d=1)).unsqueeze(1) 
    chi2 = ((times - t_weigh_avg)**2 / (sigmas**2 - s_weigh_avg**2)).sum(dim=1).unsqueeze(1)

    # nexus nodes get two [delta_T, chi2] features
    # we concat along the existing dimension to [n_sp, 2]
    data['sp'].x = cat(delta_T, chi2, dim=1) 

    return data