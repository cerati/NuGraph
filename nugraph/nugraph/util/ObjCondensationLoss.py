"""Object condensation loss function"""
import torch
from torch_scatter import scatter_max

T = torch.Tensor


class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, s_b: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.s_b = s_b
        self.q_min = q_min

    def l_b(self, f: T, f_centers: T, bkg_mask: T, n_true: int) -> T:
        """Calculate background loss term"""
        b = 1 - (f_centers.sum() / n_true)
        n_bkg = bkg_mask.sum()
        if n_bkg:
            b += (self.s_b / n_bkg) * f[bkg_mask].sum()
        return b

    def l_v(self, x: T, f: T, centers: T, e_h: T, e_p: T, n_true: int) -> T:
        """Calculate potential loss term"""
        device = x.device
        dtype = x.dtype
        n_hit = x.size(0)

        # atanh() can produce NaN/inf with mixed precision (float16/bfloat16)
        # Clamp inputs to avoid numerical issues - must clamp in float32 to avoid
        # float16 rounding issues (values get rounded to 1.0 before clamp).
        clamp_val = 0.99999
        f_clamped = f.float().clamp(-clamp_val, clamp_val).to(dtype=dtype)
        q = f_clamped.atanh().square() + self.q_min

        m_ik = torch.zeros(n_hit, n_true, dtype=torch.bool, device=device)
        m_ik[e_h, e_p] = True
        dist = (x[:, None, :] - x[centers][None, :, :]).square().sum(dim=2)
        v = torch.where(m_ik, dist, (1 - dist).clamp(0))
        v = ((v * q[centers]).sum(dim=1) * q).sum() / n_hit
        return v

    def l_p(self, f: T, bkg_mask: T, l_p: T) -> T:
        """Calculate particle loss term"""
        if l_p is None:
            dtype = f.dtype
            device = f.device
            p = torch.tensor(0., dtype=dtype, device=device)
        else:
            # Clamp to avoid NaN/inf with mixed precision - must clamp in float32
            clamp_val = 0.99999
            f_clamped = f[~bkg_mask].float().clamp(-clamp_val, clamp_val).to(dtype=dtype)
            xi = f_clamped.atanh().square()
            p = (l_p[~bkg_mask] * xi[:, None]).sum() / (xi.sum())
        return p

    def forward(self, x: T, f: T, y_i: T, y_s: T, n_true: int, e_true: T,
                l_p: T) -> T:

        device = x.device
        dtype = x.dtype

        # hit information
        n_hit = x.size(0)

        # check inputs
        if not n_true:
            return torch.zeros(3, dtype=dtype, device=device)

        # determine which hit is the condensation point for each true instance,
        # and get beta values (f_centers) and hit indices (centers)
        e_h, e_p = e_true

        # Handle empty edges case
        if e_h.size(0) == 0:
            f_centers = torch.zeros(n_true, dtype=dtype, device=device)
            centers = torch.zeros(n_true, dtype=torch.long, device=device)
        else:
            # scatter_max with the 'out' argument returns incorrect indices for empty bins.
            # The fix: don't use 'out', filter invalid indices instead.

            f_indexed = f[e_h]  # f has shape [n_hit], e_h has shape [n_edges]

            # Use n_true as dim_size to get one result per particle
            f_centers_float, centers_idx = scatter_max(f_indexed, e_p, dim_size=n_true)

            # Filter out invalid indices (>= n_edges) that scatter_max returns for empty bins
            valid_mask = centers_idx < f_indexed.size(0)
            f_centers_float = f_centers_float.clone()
            f_centers_float[~valid_mask] = 0
            centers_idx_valid = torch.where(valid_mask, centers_idx, torch.zeros_like(centers_idx))
            f_centers = f_centers_float.to(dtype=dtype)
            centers = e_h[centers_idx_valid]

        bkg_mask = (y_i == -1) & (y_s >= 0)

        # calculate loss terms
        b = self.l_b(f, f_centers, bkg_mask, n_true)
        v = self.l_v(x, f, centers, e_h, e_p, n_true)
        p = self.l_p(f, bkg_mask, l_p)

        return torch.stack([b, v, p])
