import torch

# project imports
try:
    from lib.sde_func import NeuralSDEFunc   # if your file is named sde_func.py
except ImportError:
    from lib.de_func import NeuralSDEFunc    # fallback: de_func.py

from lib.diffeq_solver import SDESolver
from lib.sde_rnn import SDE_RNN
from lib.plotting import Visualizations
from generate_timeseries import Periodic_1d

# (Optional) you can comment these two if you only save figures
import matplotlib
matplotlib.use("Qt5Agg")   # or "TkAgg" / "MacOSX" on macOS
import matplotlib.pyplot as plt


# -------------------------------
# Irregular testers (keep yours)
# -------------------------------

def apply_mask_irregular(data, keep_prob=0.5, per_feature=False, seed=None):
    """
    Per-sample missingness via mask only (timestamps remain the same).
    - data: (B, T, D)
    - keep_prob: probability to keep an observation
    - per_feature: if True, drop independently per (t,d); else per t for all features
    Returns: (masked_data, mask) with the same shape as data.
    """
    if seed is not None:
        torch.manual_seed(seed)
    B, T, D = data.shape
    if per_feature:
        mask = (torch.rand(B, T, D, device=data.device) < keep_prob).float()
    else:
        mask_t = (torch.rand(B, T, 1, device=data.device) < keep_prob).float()
        mask = mask_t.expand(B, T, D)
    data_obs = data * mask  # avoid leaking values where mask=0
    return data_obs, mask


def subsample_global_irregular(data, t, keep_prob=0.5, seed=None):
    """
    Global timestamp subsampling (same subset for all samples).
    - Keeps t[0], randomly drops others, returns a smaller time grid and sliced data.
    """
    if seed is not None:
        torch.manual_seed(seed)
    T = t.numel()
    keep = (torch.rand(T, device=t.device) < keep_prob)
    keep[0] = True                    # always keep the start
    idx = torch.where(keep)[0].sort().values
    t_sub = t.index_select(0, idx)
    data_sub = data.index_select(1, idx)
    mask_sub = torch.ones_like(data_sub)
    return data_sub, t_sub, mask_sub


# -------------------------------
# Runner
# -------------------------------

def main():
    # ------------- setup -------------
    torch.manual_seed(42)
    device = torch.device("cpu")  # change to "cuda" if available

    # ------------- build data from generate_timeseries.py -------------
    B = 8
    T = 300
    t = torch.linspace(0.0, 1.0, T, device=device)  # 1D global time grid in [0,1]

    # Use Periodic_1d from your generate_timeseries.py
    # (signature in your repo supports: sample_traj(time_steps, n_samples, noise_weight))
    gen = Periodic_1d()  # you can pass kwargs if you want to change amplitude/freq
    x = gen.sample_traj(time_steps=t, n_samples=B, noise_weight=0.02)  # -> (B, T, D)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=torch.float32)
    else:
        x = x.to(device)

    # derive shapes
    B, T, input_dim = x.shape
    mask = torch.ones_like(x)  # start with fully observed

    # ------------- optional irregular sampling -------------
    USE_MASK_IRREGULAR = True
    USE_GLOBAL_TS_SUBSAMPLE = False

    if USE_MASK_IRREGULAR:
        # per-sample missingness via mask; time grid unchanged
        x, mask = apply_mask_irregular(x, keep_prob=0.6, per_feature=False, seed=42)

    if USE_GLOBAL_TS_SUBSAMPLE:
        # global timestamp thinning; x/t/mask sliced with the same indices
        x, t, mask = subsample_global_irregular(x, t, keep_prob=0.5, seed=123)

    # ------------- build SDE-RNN -------------
    latent_dim = 16
    sde_func = NeuralSDEFunc(
        input_dim=input_dim,          # kept for API symmetry
        hidden_dim=latent_dim,
        hidden_hidden_dim=64,
        num_layers=2
    ).to(device)

    solver = SDESolver(
        input_dim=input_dim,
        sde_func=sde_func,
        method="euler",
        latents=latent_dim,
        dt=0.05,
        use_adjoint=False,
        device=device
    )

    model = SDE_RNN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        z0_diffeq_solver=solver,
        n_gru_units=100,
        n_units=100,
        obsrv_std=0.01  # smaller observation noise helps toy data learning
    ).to(device)

    # ------------- quick training loop (interpolation) -------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 500
    model.train()

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        pred, _ = model.get_reconstruction(
            time_steps_to_predict=t,
            data=x,
            truth_time_steps=t,
            mask=mask,
            n_traj_samples=1,
            mode="interp"
        )  # pred: (1,B,T,D)

        recon = pred[0]                             # (B,T,D)
        mse_mask = (mask if mask is not None else torch.ones_like(recon))
        denom = mse_mask.sum().clamp_min(1.0)
        loss = ((recon - x) ** 2 * mse_mask).sum() / denom

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[epoch {epoch:03d}] train MSE = {loss.item():.6f}")

    # ------------- evaluation + plotting -------------
    model.eval()
    with torch.no_grad():
        pred, info = model.get_reconstruction(
            time_steps_to_predict=t,
            data=x,
            truth_time_steps=t,
            mask=mask,
            n_traj_samples=1,
            mode="interp"
        )
    print("prediction shape =", tuple(pred.shape))  # (1, B, T_kept, D)

    recon = pred[0]
    mse_mask = (mask if mask is not None else torch.ones_like(recon))
    denom = mse_mask.sum().clamp_min(1.0)
    final_mse = ((recon - x) ** 2 * mse_mask).sum() / denom
    print(f"final masked MSE = {final_mse.item():.6f}")

    # build data_dict for plotting (detach to avoid numpy() grad errors)
    data_dict = {
        "observed_data": x.detach(),
        "observed_tp": t.detach(),
        "observed_mask": (mask.detach() if mask is not None else None),
        "data_to_predict": x.detach(),
        "tp_to_predict": t.detach(),
        "mask_predicted_data": (mask.detach() if mask is not None else None),
    }

    viz = Visualizations(device)
    viz.draw_all_plots_one_dim(
        data_dict=data_dict,
        model=model,
        plot_name="toy_recon.png",
        save=True,             # saves under plots/<experimentID>/
        experimentID="toy"
    )
    print("Saved plots to: plots/toy/toy_recon.png")


if __name__ == "__main__":
    main()
