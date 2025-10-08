import torch
import torch.nn as nn
import lib.utils as utils

class LipSwish(nn.Module):
	def forward(self, x):
		return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
	def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation='lipswish'):
		super().__init__()

		if activation == 'lipswish':
			activation_fn = LipSwish()
		else:
			activation_fn = nn.ReLU()

		msdel = [nn.Linear(in_size, hidden_dim), activation_fn]
		for _ in range(num_layers - 1):
			msdel.append(nn.Linear(hidden_dim, hidden_dim))
			msdel.append(activation_fn)
		msdel.append(nn.Linear(hidden_dim, out_size))
		if tanh:
			msdel.append(nn.Tanh())
		self._msdel = nn.Sequential(*msdel)

	def forward(self, x):
		return self._msdel(x)


class NeuralSDEFunc(nn.Module):
	def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
		super(NeuralSDEFunc, self).__init__()
		self.sde_type = "ito"
		self.noise_type = "diagonal"  # or "scalar"

		self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
		self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
		self.linear_out = nn.Linear(hidden_dim, hidden_dim)
		self.noise_in = nn.Linear(hidden_dim + 1, hidden_dim)
		self.g_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)

	# Optional convenience: calling the module uses drift by default
	def forward(self, t, y):
		return self.f(t, y)

	def set_X(self, coeffs, times):
		"""Optional: set a control path if you use CDE; comment out if unused."""
		self.coeffs = coeffs
		self.times = times

	# self.X = torchcde.CubicSpline(self.coeffs, self.times)

	def _broadcast_t_like_y(self, t, y):
		"""
        Make t a column tensor with the same batch/time shape as y[..., :1].
        Works for y shapes (B,H) or (B,T,H).
        """
		if torch.is_tensor(t):
			t_val = float(t.reshape(-1)[0].item())
		else:
			t_val = float(t)
		return torch.full_like(y[..., :1], t_val)

	def f(self, t, y):
		t_col = self._broadcast_t_like_y(t, y)  # (B,1) or (B,T,1)
		yy = self.linear_in(torch.cat((t_col, y), dim=-1))
		return self.f_net(yy)

	def g(self, t, y):
		t_col = self._broadcast_t_like_y(t, y)  # (B,1) or (B,T,1)
		yy = self.noise_in(torch.cat((t_col, y), dim=-1))
		return self.g_net(yy)


class NeuralSDEFunc_w_Poisson(NeuralSDEFunc):

	def __init__(self, input_dim, latent_dim, sde_func_net,
				 lambda_net, device=torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for SDE. Analog of a continous latent state
		"""
		super(NeuralSDEFunc_w_Poisson, self).__init__(input_dim, latent_dim, sde_func_net, device)

		self.latent_sde = NeuralSDEFunc(input_dim=input_dim,
										hidden_dim=latent_dim,
										sde_func_net=sde_func_net,
										device=device)

		self.latent_dim = latent_dim
		self.lambda_net = lambda_net
		# The computation of poisson likelihood can become numerically unstable.
		# The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
		# Exponent of lambda can also take large values
		# So we divide lambda by the constant and then multiply the integral of lambda by the constant
		self.const_for_lambda = torch.Tensor([100.]).to(device)

	def extract_poisson_rate(self, augmented, final_result=True):
		y, log_lambdas, int_lambda = None, None, None

		assert (augmented.size(-1) == self.latent_dim + self.input_dim)
		latent_lam_dim = self.latent_dim // 2

		if len(augmented.size()) == 3:
			int_lambda = augmented[:, :, -self.input_dim:]
			y_latent_lam = augmented[:, :, :-self.input_dim]

			log_lambdas = self.lambda_net(y_latent_lam[:, :, -latent_lam_dim:])
			y = y_latent_lam[:, :, :-latent_lam_dim]

		elif len(augmented.size()) == 4:
			int_lambda = augmented[:, :, :, -self.input_dim:]
			y_latent_lam = augmented[:, :, :, :-self.input_dim]

			log_lambdas = self.lambda_net(y_latent_lam[:, :, :, -latent_lam_dim:])
			y = y_latent_lam[:, :, :, :-latent_lam_dim]

		# Multiply the intergral over lambda by a constant
		# only when we have finished the integral computation (i.e. this is not a call in get_sde_gradient_nn)
		if final_result:
			int_lambda = int_lambda * self.const_for_lambda

		# Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
		assert (y.size(-1) == latent_lam_dim)

		return y, log_lambdas, int_lambda, y_latent_lam

	def get_sde_gradient_nn(self, t_local, augmented):
		y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result=False)
		dydt_dldt = self.latent_sde.f(t_local, y_latent_lam)

		log_lam = log_lam - torch.log(self.const_for_lambda)

		return torch.cat((dydt_dldt, torch.exp(log_lam)), dim=-1)


class ODEFunc(nn.Module):
	def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc, self).__init__()

		self.input_dim = input_dim
		self.device = device

		utils.init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards=False):
		"""
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)


#####################################################################################################

class ODEFunc_w_Poisson(ODEFunc):

	def __init__(self, input_dim, latent_dim, ode_func_net,
				 lambda_net, device=torch.device("cpu")):
		"""
		input_dim: dimensionality of the input
		latent_dim: dimensionality used for ODE. Analog of a continous latent state
		"""
		super(ODEFunc_w_Poisson, self).__init__(input_dim, latent_dim, ode_func_net, device)

		self.latent_ode = ODEFunc(input_dim=input_dim,
								  latent_dim=latent_dim,
								  ode_func_net=ode_func_net,
								  device=device)

		self.latent_dim = latent_dim
		self.lambda_net = lambda_net
		# The computation of poisson likelihood can become numerically unstable.
		# The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
		# Exponent of lambda can also take large values
		# So we divide lambda by the constant and then multiply the integral of lambda by the constant
		self.const_for_lambda = torch.Tensor([100.]).to(device)

	def extract_poisson_rate(self, augmented, final_result=True):
		y, log_lambdas, int_lambda = None, None, None

		assert (augmented.size(-1) == self.latent_dim + self.input_dim)
		latent_lam_dim = self.latent_dim // 2

		if len(augmented.size()) == 3:
			int_lambda = augmented[:, :, -self.input_dim:]
			y_latent_lam = augmented[:, :, :-self.input_dim]

			log_lambdas = self.lambda_net(y_latent_lam[:, :, -latent_lam_dim:])
			y = y_latent_lam[:, :, :-latent_lam_dim]

		elif len(augmented.size()) == 4:
			int_lambda = augmented[:, :, :, -self.input_dim:]
			y_latent_lam = augmented[:, :, :, :-self.input_dim]

			log_lambdas = self.lambda_net(y_latent_lam[:, :, :, -latent_lam_dim:])
			y = y_latent_lam[:, :, :, :-latent_lam_dim]

		# Multiply the intergral over lambda by a constant
		# only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
		if final_result:
			int_lambda = int_lambda * self.const_for_lambda

		# Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
		assert (y.size(-1) == latent_lam_dim)

		return y, log_lambdas, int_lambda, y_latent_lam

	def get_ode_gradient_nn(self, t_local, augmented):
		y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result=False)
		dydt_dldt = self.latent_ode(t_local, y_latent_lam)

		log_lam = log_lam - torch.log(self.const_for_lambda)
		return torch.cat((dydt_dldt, torch.exp(log_lam)), -1)
