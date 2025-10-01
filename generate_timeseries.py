###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
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


def add_noise_sde(
		traj_list: torch.Tensor,
		time_steps,
		noise_type: str = "ou",  # "ou" 或 "bm"
		sigma: float = 0.1,
		ou_theta: float = 1.5,
		ou_mu: float = 0.0,
) -> torch.Tensor:
	"""用 SDE 生成的路径来给 value 通道加噪（不动 time）。
	traj_list: (B, T, D) 张量（D 为值维度）
	time_steps: (T,) numpy or torch
	"""
	assert isinstance(traj_list, torch.Tensor), "traj_list must be torch.Tensor"
	device = traj_list.device
	dtype = traj_list.dtype

	if isinstance(time_steps, torch.Tensor):
		t = time_steps.to(device=device, dtype=dtype)
	else:
		t = torch.as_tensor(time_steps, device=device, dtype=dtype)

	B, T, D = traj_list.shape
	assert T == t.numel(), "time_steps length must match traj length"

	dt = t[1:] - t[:-1]  # (T-1,)
	dW = torch.randn(B, T - 1, D, device=device, dtype=dtype) * torch.sqrt(
		dt.view(1, -1, 1).clamp_min(1e-12)
	)

	nt = noise_type.lower()
	device = traj_list.device
	dtype = traj_list.dtype  # <- usually float32
	B, T, D = traj_list.shape
	dt = (time_steps[1:] - time_steps[:-1]).to(device=device, dtype=dtype)
	noise = torch.zeros(B, T, D, device=device, dtype=dtype)  # <- dtype match

	if nt == "bm":
		# Brownian: X_k = X_{k-1} + sigma dW
		for k in range(1, T):
			noise[:, k, :] = noise[:, k - 1, :] + sigma * dW[:, k - 1, :]
	elif nt == "ou":
		# Ornstein–Uhlenbeck: X_k = X_{k-1} + theta (mu - X_{k-1}) dt + sigma dW
		for k in range(1, T):
			prev = noise[:, k - 1, :]
			drift = ou_theta * (ou_mu - prev) * dt[k - 1]
			noise[:, k, :] = prev + drift + sigma * dW[:, k - 1, :]
	else:
		raise ValueError(f"Unknown noise_type '{noise_type}'; use 'ou' or 'bm'.")

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
		# ---- 统一与 run_models.py 的命名 ----
		use_sde_noise: bool = False,
		sde_noise_type: str = "ou",     # "ou" / "bm"
		sde_noise_type: str | None = None,
		sde_noise_sigma: float = 0.1,
		sde_ou_theta: float = 1.5,
		sde_ou_mu: float = 0.0,
	):
		"""

		:type sde_noise_type: str | None
		"""
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
			# 只要 value 列（去掉 time），再加 batch 维
			traj = np.expand_dims(traj[:, 1:], 0)   # (1, T, 1)
			traj_list.append(traj)

		traj_list = np.array(traj_list)  # (B, 1, T, 1)
		traj_list = torch.tensor(traj_list, device=self.device, dtype=torch.float32).squeeze(1)  # -> (B, T, 1)

		if sde_noise_type is None:
			# use the original i.i.d. noise
			traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		else:
			# use your SDE noise adder; e.g. add_noise_sde(traj_list, time_steps, ...)
			traj_list = add_noise_sde(
				traj_list,
				time_steps=time_steps,
				noise_type=sde_noise_type,
				sigma=sde_noise_sigma,
				ou_theta=sde_ou_theta,
				ou_mu=sde_ou_mu,
			)
		return traj_list


