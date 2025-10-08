###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
from torch import Tensor

if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point, 
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise

	@staticmethod
	def add_noise_sde(
			traj_list: torch.Tensor,
			time_steps,
			noise_type: str = "ou",  # "ou" or "bm"
			sigma: float = 0.1,  # diffusion strength
			ou_theta: float = 1.5,  # OU mean-reversion rate
			ou_mu: float = 0.0,  # OU long-run mean
	) -> torch.Tensor:
		"""
        Add SDE-driven noise to the *value* channel(s) of a trajectory.
        - traj_list: (B, T, D) torch tensor (D = number of value dims)
        - time_steps: (T,) numpy array or torch tensor of strictly increasing times
        Returns: noisy trajectory (B, T, D) on the same device/dtype as traj_list.

        Noise models:
          * bm: X_k = X_{k-1}            + sigma * dW
          * ou: X_k = X_{k-1} + theta*(mu - X_{k-1})*dt + sigma * dW
        """
		if not isinstance(traj_list, torch.Tensor):
			raise TypeError("traj_list must be a torch.Tensor")
		device = traj_list.device
		dtype = traj_list.dtype

		# Make time a torch tensor on the same device/dtype
		t = torch.as_tensor(time_steps, device=device, dtype=dtype)
		if t.ndim != 1:
			raise ValueError("time_steps must be 1D of shape (T,).")
		B, T, D = traj_list.shape
		if T != t.numel():
			raise ValueError(f"Mismatch: traj length T={T} vs len(time_steps)={t.numel()}.")

		# Time increments and Wiener increments
		dt = (t[1:] - t[:-1]).clamp_min(1e-12)  # (T-1,)
		dW = torch.randn(B, T - 1, D, device=device, dtype=dtype) * dt.view(1, -1, 1).sqrt()

		# Simulate noise path in value space; start at zero
		noise = torch.zeros(B, T, D, device=device, dtype=dtype)
		nt = (noise_type or "").lower()

		if nt == "bm":
			# Brownian increments
			for k in range(1, T):
				noise[:, k, :] = noise[:, k - 1, :] + sigma * dW[:, k - 1, :]

		elif nt == "ou":
			# Ornstein–Uhlenbeck: mean-reverting around ou_mu
			for k in range(1, T):
				prev = noise[:, k - 1, :]
				drift = ou_theta * (ou_mu - prev) * dt[k - 1]  # deterministic pull
				noise[:, k, :] = prev + drift + sigma * dW[:, k - 1, :]

		else:
			raise ValueError(f"Unknown noise_type '{noise_type}'. Use 'ou' or 'bm'.")

		# Add value noise, keep time as-is (time is not part of traj_list here)
		return traj_list + noise


class Periodic_1d(TimeSeries):
	def __init__(self, device=torch.device("cpu"),
				 init_freq=0.3, init_amplitude=1.,
				 final_amplitude=10., final_freq=1.,
				 z0=0.):
		super(Periodic_1d, self).__init__(device)
		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(
			self,
			time_steps,
			n_samples: int = 1,
			noise_weight: float = 1.0,
			# choose SDE noise (else i.i.d. noise)
			use_sde_noise: bool = False,
			sde_noise_type: str = "ou",  # "ou" or "bm"
			sde_noise_sigma: float = 0.1,
			sde_ou_theta: float = 1.5,
			sde_ou_mu: float = 0.0,
	):
		# -------- generate clean periodic data --------
		traj_list = []
		for _ in range(n_samples):
			init_freq = assign_value_or_sample(self.init_freq, [0.4, 0.8])
			final_freq = init_freq if (self.final_freq is None) else assign_value_or_sample(self.final_freq, [0.4, 0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0., 1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0., 1.])
			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(
				time_steps,
				init_freq=init_freq,
				init_amplitude=init_amplitude,
				starting_point=noisy_z0,
				final_amplitude=final_amplitude,
				final_freq=final_freq
			)
			# keep only the value column (drop time) and add batch dim
			traj = np.expand_dims(traj[:, 1:], 0)  # (1, T, 1)
			traj_list.append(traj)

		# -> (B, T, 1) torch.float32 on self.device
		traj_list = np.array(traj_list)
		traj_list = torch.tensor(traj_list, device=self.device, dtype=torch.float32).squeeze(1)

		# -------- add noise (choose exactly one) --------
		if use_sde_noise and (sde_noise_type is not None):
			# SDE noise: Brownian ("bm") or Ornstein–Uhlenbeck ("ou")
			traj_list = TimeSeries.add_noise_sde(
				traj_list,
				time_steps=time_steps,
				noise_type=sde_noise_type,
				sigma=sde_noise_sigma,
				ou_theta=sde_ou_theta,
				ou_mu=sde_ou_mu,
			)
		else:
			# i.i.d. noise on values (skip if weight == 0)
			if noise_weight is not None and float(noise_weight) != 0.0:
				traj_list = self.add_noise(traj_list, time_steps, noise_weight)

		return traj_list


