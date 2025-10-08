import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline


class SDE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device=torch.device("cpu"),
                 z0_diffeq_solver=None, n_gru_units=100, n_units=100,
                 concat_mask=False, obsrv_std=0.1,
                 use_binary_classif=False, classif_per_tp=False,
                 n_labels=1, train_classif_w_reconstr=False):
        super().__init__(input_dim, latent_dim, device,
                         obsrv_std, use_binary_classif,
                         classif_per_tp, use_poisson_proc=False,
                         linear_classifier=False, n_labels=n_labels,
                         train_classif_w_reconstr=train_classif_w_reconstr)

        self.sde_gru = Encoder_z0_SDE_RNN(
            latent_dim=latent_dim,
            input_dim=input_dim * 2,         # data + mask
            z0_diffeq_solver=z0_diffeq_solver,
            n_gru_units=n_gru_units, device=device).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim)
        )
        utils.init_network_weights(self.decoder)

        # --- expose solver & meta for plotting compatibility ---
        self.z0_diffeq_solver = z0_diffeq_solver
        self.diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim

    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps,
                           mask=None, n_traj_samples=None, mode=None):
        # 1) Interp-only guard
        if (len(truth_time_steps) != len(time_steps_to_predict)) or (
                torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for SDE-RNN")  # interp only

        assert len(truth_time_steps) == len(time_steps_to_predict)
        assert mask is not None

        # 2) Concatenate data + mask for the encoder
        data_and_mask = torch.cat([data, mask], dim=-1)

        # 3) Run the SDE-GRU encoder over the timeline (no backwards run for interpolation)
        # Returns: last_y, last_std, latent_ys, extra_info_enc
        last_y, last_std, latent_ys, extra_info_enc = self.sde_gru.run_sdernn(
            data_and_mask, truth_time_steps, run_backwards=False
        )

        # latent_ys: [1, T, B, latent_dim] -> [1, B, T, latent_dim]
        latent_ys = latent_ys.permute(0, 2, 1, 3)
        last_hidden = latent_ys[:, :, -1, :]  # [1, B, latent_dim]

        # 4) Decode to data space
        outputs = self.decoder(latent_ys)  # [1, B, T, D]

        # 5) Shift outputs so prediction at step k is compared to true x_k (teacher forcing)
        first_point = data[:, 0, :]  # [B, D]
        outputs = utils.shift_outputs(outputs, first_point)  # [1, B, T, D]

        # 6) Extra info for downstream losses / classification
        extra_info = {
            # these three are a placeholder tuple matching downstream expectations
            "first_point": (last_hidden, 0.0, last_hidden)  # (mu, std, sample) shape: [1, B, L]
        }

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(latent_ys)  # [1, B, T, n_labels]
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)  # [1, B] or [1, B, n_labels]

        info = {}
        info["first_point"] = (last_y, last_std, extra_info_enc)

        # 关键：把潜在路径放进去，plotting 会自动识别
        # 形状可以是 (B, T, D) 或 (S, B, T, D)
        info["latent_traj"] = latent_ys

        # ---- add this block right before:  return outputs, info ----
        if self.use_binary_classif:
            if self.classif_per_tp:
                # time-wise classification  -> expect [B, T, ...]
                logits = self.classifier(latent_ys.squeeze(0))  # [B, T, n_labels] or [B, T, 1]
            else:
                # sequence-wise classification -> expect [B, ...]
                logits = self.classifier(last_hidden.squeeze(0))  # [B, n_labels] or [B, 1]

            info["label_predictions"] = logits.squeeze(-1)  # make it [B] or [B, T]

        return outputs, info

    # def compute_all_losses(self, batch_dict, n_tp_to_sample=None, n_traj_samples=1, kl_coef=1.):
    #     # 1) forward
    #     pred_x, info = self.get_reconstruction(
    #         batch_dict["tp_to_predict"],
    #         batch_dict["observed_data"],
    #         batch_dict["observed_tp"],
    #         mask=batch_dict.get("observed_mask", None),
    #         n_traj_samples=n_traj_samples,
    #         mode=batch_dict["mode"]
    #     )
    #     # 2) likelihood / mse （与 Baseline 相同）
    #     likelihood = self.get_gaussian_likelihood(batch_dict["data_to_predict"], pred_x,
    #                                               mask=batch_dict.get("mask_predicted_data", None))
    #     mse = self.get_mse(batch_dict["data_to_predict"], pred_x,
    #                        mask=batch_dict.get("mask_predicted_data", None))
    #
    #     # 3) optional CE (classification)
    #     ce_loss = torch.tensor(0., device=pred_x.device)
    #     if self.use_binary_classif and (batch_dict["labels"] is not None):
    #         if batch_dict["labels"].size(-1) == 1 or len(batch_dict["labels"].size()) == 1:
    #             ce_loss = compute_binary_CE_loss(info["label_predictions"], batch_dict["labels"])
    #         else:
    #             ce_loss = compute_multiclass_CE_loss(info["label_predictions"], batch_dict["labels"],
    #                                                  mask=batch_dict.get("mask_predicted_data", None))
    #
    #     # 4) optional Poisson proc likelihood
    #     pois_log_likelihood = torch.tensor(0., device=pred_x.device)
    #     if self.use_poisson_proc:
    #         pois_log_likelihood = compute_poisson_proc_likelihood(
    #             batch_dict["data_to_predict"], pred_x, info,
    #             mask=batch_dict.get("mask_predicted_data", None)
    #         ).mean(dim=1)  # mean over n_traj
    #
    #     # 5) final loss 与 Baseline 对齐
    #     loss = - torch.mean(likelihood)
    #     if self.use_poisson_proc:
    #         loss = loss - 0.1 * pois_log_likelihood
    #     if self.use_binary_classif:
    #         loss = ce_loss if not self.train_classif_w_reconstr else (loss + 100 * ce_loss)
    #
    #     # 6) 汇总与 Baseline 完全同名的键
    #     results = {
    #         "loss": torch.mean(loss),
    #         "likelihood": torch.mean(likelihood).detach(),
    #         "mse": torch.mean(mse).detach(),
    #         "pois_likelihood": torch.mean(pois_log_likelihood).detach(),
    #         "ce_loss": torch.mean(ce_loss).detach(),
    #         "kl": 0., "kl_first_p": 0., "std_first_p": 0.,
    #     }
    #     if batch_dict["labels"] is not None and self.use_binary_classif:
    #         results["label_predictions"] = info["label_predictions"].detach()
    #     return results
