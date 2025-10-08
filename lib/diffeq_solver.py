import time
import numpy as np

import torch
import torch.nn as nn
import torchsde
import lib.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint

#####################################################################################################

class SDESolver(nn.Module):
    """
    Thin wrapper around torchsde.sdeint that:
      - accepts y0 as (B,D) or (S,B,D) where S is MC samples
      - flattens to (S*B, D) for integration
      - restores to (S,B,T,D) on return
      - builds BrownianPath from w0 so device/dtype are inherited
    """
    def __init__(self, input_dim, sde_func, method, latents,
                 dt=0.05, use_adjoint=True, device=torch.device("cpu")):
        super().__init__()
        self.sde_method = method
        self.latents = latents
        self.device = device
        self.sde_func = sde_func
        self.dt = dt
        self.sdeint = torchsde.sdeint_adjoint if use_adjoint else torchsde.sdeint

    def _make_bm(self, y0, ts):
        """Create BrownianPath; device/dtype come from w0 (zeros_like y0)."""
        t0 = float(ts[0].item() if torch.is_tensor(ts) else ts)
        w0 = torch.zeros_like(y0)
        return torchsde.BrownianPath(t0=t0, w0=w0)

    def _flatten_y0(self, y0):
        """
        y0: (B,D) or (S,B,D) -> (batch, D)
        returns: y0_flat, S, B, D
        """
        if y0.dim() == 2:
            B, D = y0.size()
            S = 1
            y0_flat = y0
        elif y0.dim() == 3:
            S, B, D = y0.size()
            y0_flat = y0.reshape(S * B, D)
        else:
            raise ValueError("y0 must be (B,D) or (S,B,D)")
        return y0_flat, S, B, D

    def _restore_path(self, path, S, B, D):
        """
        path: (T, batch, D) -> (S,B,T,D)
        """
        T = path.size(0)
        if S == 1:
            return path.permute(1, 0, 2).unsqueeze(0).contiguous()
        else:
            return path.view(T, S, B, D).permute(1, 2, 0, 3).contiguous()

    def _prepare_ts(self, ts_in, ref_tensor):
        """
        ts_in: arbitrary order, may contain ties. Shape (T,) or (T,1) etc.
        ref_tensor: for device/dtype
        returns:
          ts_inc: strictly increasing (T,)
          inv_idx: to map results back to original order (T,)
        """
        ts = ts_in.to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(-1)

        # sort ascending and remember index
        ts_sorted, sort_idx = torch.sort(ts)

        # enforce strictly increasing by nudging ties with a tiny epsilon
        eps = torch.finfo(ts_sorted.dtype).eps * 1000
        ts_inc = ts_sorted.clone()
        for i in range(1, ts_inc.numel()):
            if ts_inc[i] <= ts_inc[i - 1]:
                ts_inc[i] = ts_inc[i - 1] + eps

        # build inverse index to restore original order:
        inv_idx = torch.empty_like(sort_idx)
        inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

        return ts_inc, inv_idx

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        Integrate on a sanitized increasing grid, then reorder to the caller's order.
        first_point: (B,D) or (S,B,D)
        time_steps_to_predict: (T,)
        returns: (S,B,T,D) aligned with the ORIGINAL order of time_steps_to_predict
        """
        y0_flat, S, B, D = self._flatten_y0(first_point)

        ts_inc, inv_idx = self._prepare_ts(time_steps_to_predict, y0_flat)

        bm = self._make_bm(y0_flat, ts_inc)

        path_sorted = self.sdeint(self.sde_func, y0_flat, ts_inc,
                                  method=self.sde_method, dt=self.dt, bm=bm)
        path = path_sorted.index_select(0, inv_idx)
        pred_y = self._restore_path(path, S, B, D)

        first_idx = 0
        err0 = (path_sorted[0] - y0_flat).abs().max().item()
        assert err0 < 5e-2, f"Initial mismatch too large: {err0:.3e}"

        return pred_y

    @torch.no_grad()
    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples=1):
        x = starting_point_enc
        if x.dim() == 2:
            B, D = x.size()
            x = x.unsqueeze(0).expand(n_traj_samples, B, D).contiguous()
        elif x.dim() == 3:
            if x.size(0) == 1 and n_traj_samples > 1:
                x = x.expand(n_traj_samples, -1, -1).contiguous()
        else:
            raise ValueError("starting_point_enc must be (B,D) or (S,B,D)")

        y0_flat, S, B, D = self._flatten_y0(x)

        # sanitize times and integrate
        ts_inc, inv_idx = self._prepare_ts(time_steps_to_predict, y0_flat)
        bm = self._make_bm(y0_flat, ts_inc)
        path_sorted = self.sdeint(self.sde_func, y0_flat, ts_inc,
                                  method=self.sde_method, dt=self.dt, bm=bm)
        path = path_sorted.index_select(0, inv_idx)

        return self._restore_path(path, S, B, D)


class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents,
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y