###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.distributions import Normal

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.sde_rnn import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.de_func import ODEFunc, ODEFunc_w_Poisson, NeuralSDEFunc, NeuralSDEFunc_w_Poisson
from lib.diffeq_solver import SDESolver, DiffeqSolver
from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--sde-rnn', action='store_true', help="Run SDE-RNN baseline (GRU updates + SDE evolution).")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

parser.add_argument("--plot", action="store_true", help="Save reconstruction plots for one test batch.")
parser.add_argument("--plot-k", type=int, default=1, help="Number of test batches to plot (default 1).")

# ---- SDE-driven noise on toy/periodic datasets ----
parser.add_argument("--use-sde-noise", action="store_true", help="If set, use SDE-driven noise for toy/periodic data (instead of old Gaussian jitter).")
parser.add_argument("--sde-noise-type", type=str, default="ou", choices=["ou", "bm"], help="Type of SDE noise: 'ou' (Ornstein–Uhlenbeck) or 'bm' (Brownian).")
parser.add_argument("--sde-sigma", type=float, default=0.10, help="Diffusion scale sigma for the SDE noise.")
parser.add_argument("--ou-theta", type=float, default=1.50, help="OU mean-reversion speed (only used when --sde-noise-type=ou).")
parser.add_argument("--ou-mu", type=float, default=0.0, help="OU long-term mean (only used when --sde-noise-type=ou).")

parser.add_argument("--miss-rate", type=float, default=1.0, help="Fraction of values kept as observed (0~1). 1.0 = no missingness")
parser.add_argument("--miss-scheme", type=str, default="per-value", choices=["per-value", "per-time", "per-dim"], help="How to drop observations: per-value (iid), per-time (entire timestep), per-dim (entire variable)")

args = parser.parse_args()
file_name = os.path.basename(__file__)[:-3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
	# ----------------- seeds -----------------
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	# ----------------- experiment ids & paths -----------------
	utils.makedirs(args.save)
	experimentID = args.load if args.load is not None else int(SystemRandom().random() * 100000)
	ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")
	file_name = os.path.basename(__file__)[:-3]
	input_command = " ".join(sys.argv)
	# ----------------- data -----------------
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]
	classif_per_tp = bool(data_obj.get("classif_per_tp", False))
	if args.classif and (args.dataset in ("hopper", "periodic")):
		raise Exception("Classification task is not available for MuJoCo and 1d datasets")

	# Always define n_labels so regression paths won't crash.
	n_labels = int(data_obj.get("n_labels", 1))
	# ----------------- build model -----------------
	obsrv_std = torch.tensor([1e-3 if args.dataset == "hopper" else 1e-2], device=device)
	z0_prior = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

	if args.rnn_vae:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents,
						device=device,
						rec_dims=args.rec_dims,
						concat_mask=True,
						obsrv_std=obsrv_std,
						z0_prior=z0_prior,
						use_binary_classif=args.classif,
						classif_per_tp=classif_per_tp,
						linear_classifier=args.linear_classif,
						n_units=args.units,
						input_space_decay=args.input_decay,
						cell=args.rnn_cell,
						n_labels=n_labels,
						train_classif_w_reconstr=(args.dataset == "physionet")
						).to(device)



	elif args.classic_rnn:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device,
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.ode_rnn:
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents

		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims,
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim,
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents,
										odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

		model = ODE_RNN(input_dim, n_ode_gru_dims, device = device,
			z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.sde_rnn:
		# Build SDE func & solver (align with ODE-RNN interface)
		from lib.de_func import NeuralSDEFunc
		from lib.diffeq_solver import SDESolver

		n_sde_gru_dims = args.latents

		sde_func = NeuralSDEFunc(
			input_dim=input_dim,
			hidden_dim=n_sde_gru_dims,
			hidden_hidden_dim=args.units,
			num_layers=args.gen_layers
		).to(device)

		z0_diffeq_solver = SDESolver(
			input_dim=input_dim,
			sde_func=sde_func,
			method="euler",  # or "milstein"
			latents=n_sde_gru_dims,
			dt=0.05,
			use_adjoint=False,
			device=device
		)

		model = SDE_RNN(
			input_dim=input_dim,
			latent_dim=n_sde_gru_dims,
			device=device,
			z0_diffeq_solver=z0_diffeq_solver,
			n_gru_units=args.gru_units,
			concat_mask=True,  # keep parity with ODE-RNN / Classic RNN
			obsrv_std=obsrv_std,
			n_units=args.units,
			use_binary_classif=args.classif,
			classif_per_tp=classif_per_tp,
			n_labels=n_labels,
			train_classif_w_reconstr=(args.dataset == "physionet")
		).to(device)


	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	# === Best checkpoint tracker (before training loop) ===
	best_dir = os.path.join(args.save, "checkpoints")
	os.makedirs(best_dir, exist_ok=True)
	best_path = os.path.join(best_dir, f"{args.dataset}_sdernn_best.pt")
	select_metric = "loss"  # or "mse"/"auc" as you prefer
	best_val = float("inf")  # use -inf if your metric is "higher is better"
	# =======================================================

##################################################################

	if args.viz:
		viz = Visualizations(device)

	##################################################################

	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)
		exit()

	##################################################################
	# Training

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]

	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			with torch.no_grad():

				test_res = compute_loss_all_batches(model,
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches,
					test_res["loss"].detach(), test_res["likelihood"].detach(),
					test_res["kl_first_p"], test_res["std_first_p"])

				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))
				# === Save best checkpoint based on a validation metric ===
				select_metric = "loss"  # or "mse" / "auc" / "accuracy"
				higher_is_better = select_metric in ("auc", "accuracy")
				val_value = float(test_res.get(select_metric, test_res.get("mse", test_res.get("loss", 1e9))))
				is_better = (val_value > best_val) if higher_is_better else (val_value < best_val)
				if is_better:
					best_val = val_value
					torch.save({'args': args, 'state_dict': model.state_dict()}, best_path)
					print(f"[checkpoint] new best {select_metric}={best_val:.6f} -> {best_path}")

			torch.save({
				'args': args,
				'state_dict': model.state_dict(),
			}, ckpt_path)


			# Plotting
			if args.viz:
				with torch.no_grad():
					batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
					data_dict = {
						"observed_data": batch_dict["observed_data"],  # (B,T,D)
						"observed_tp": batch_dict["observed_tp"],  # (T,)
						"observed_mask": batch_dict.get("observed_mask", None),  # (B,T,D) or None
						"data_to_predict": batch_dict["data_to_predict"],  # (B,T,D)
						"tp_to_predict": batch_dict["tp_to_predict"],  # (T,)
						"mask_predicted_data": batch_dict.get("mask_predicted_data", None),
					}
					# viz = Visualizations(device)
					# plot_id = itr // num_batches // n_iters_to_viz
					# viz.draw_all_plots_one_dim(
					# 	data_dict=data_dict,  # <-- dict with tensors
					# 	model=model,
					# 	plot_name=file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",  # e.g. f"epoch_{epoch:04d}.png"
					# 	save=True,
					# 	experimentID=args.dataset if hasattr(args, "dataset") else "default"
					# )

					if args.dataset == "periodic":
						plot_id = itr // num_batches // n_iters_to_viz
						viz.draw_all_plots_one_dim(data_dict, model,
							plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
							experimentID = experimentID, save=True)
						plt.pause(0.01)
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)

# ----------------- Final plotting (optional) -----------------
# if args.plot_final:
# 	# make sure best_path exists in this scope (same as where you set it before training)
# 	# e.g., define best_dir/best_path once, before training:
# 	# best_dir = os.path.join(args.save, "checkpoints")
# 	# os.makedirs(best_dir, exist_ok=True)
# 	# best_path = os.path.join(best_dir, f"{args.dataset}_sdernn_best.pt")
#
# 	# load best if present, else use the last model
# 	if os.path.exists(best_path):
# 		ckpt = torch.load(best_path, map_location=device)
# 		model.load_state_dict(ckpt["state_dict"])
# 		print(f"[final] Loaded best checkpoint from: {best_path}")
# 	else:
# 		print(f"[final] Best checkpoint not found at {best_path}; using last model.")
# 	model.eval()
#
# 	from lib.plotting import Visualizations
# 	viz = Visualizations(device)
#
# 	plotted = 0
# 	with torch.no_grad():
# 		for _ in range(args.plot_k):
# 			# your data loaders live in data_obj; it’s an infinite generator
# 			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
# 			data_dict = {
# 				"observed_data": batch_dict["observed_data"],
# 				"observed_tp": batch_dict["observed_tp"],
# 				"observed_mask": batch_dict.get("observed_mask", None),
# 				"data_to_predict": batch_dict["data_to_predict"],
# 				"tp_to_predict": batch_dict["tp_to_predict"],
# 				"mask_predicted_data": batch_dict.get("mask_predicted_data", None),
# 			}
# 			plot_name = f"{args.dataset}_final_{plotted}.png"
# 			viz.draw_all_plots_one_dim(
# 				data_dict=data_dict, model=model,
# 				plot_name=plot_name, save=True,
# 				experimentID=f"{args.dataset}_final"
# 			)
# 			print(f"[final] Saved plot: plots/{args.dataset}_final/{plot_name}")
# 			plotted += 1

