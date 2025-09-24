import torch

# project imports
from lib.de_func import NeuralSDEFunc
from lib.diffeq_solver import SDESolver
from lib.sde_rnn import SDE_RNN
from lib.plotting import Visualizations
import matplotlib
matplotlib.use("Qt5Agg")   # or "TkAgg" / "MacOSX" (mac only)
import matplotlib.pyplot as plt

# -------------------------------
# Toyset generators
# -------------------------------

def toy_ou(B=8, T=200, theta=1.0, mu=0.0, sigma=0.3, x0=0.0, device="cpu"):
    """1D Ornstein–Uhlenbeck (mean-reverting)"""
    t = torch.linspace(0, 2.0, T, device=device)
    dt = t[1] - t[0]
    x = torch.zeros(B, T, 1, device=device)
    x[:, 0, 0] = x0
    for k in range(1, T):
        dw = torch.randn(B, 1, device=device) * torch.sqrt(dt)
        x[:, k, 0] = x[:, k-1, 0] + theta*(mu - x[:, k-1, 0])*dt + sigma*dw[:, 0]
    mask = torch.ones_like(x)
    return x, t, mask

def toy_spiral(B=8, T=300, omega=4.0, decay=0.2, sigma=0.05, device="cpu"):
    """2D rotating spiral with diffusion"""
    t = torch.linspace(0, 3.0, T, device=device)
    dt = t[1] - t[0]
    x = torch.zeros(B, T, 2, device=device)
    x[:, 0, :] = torch.randn(B, 2, device=device) * 0.3
    A = torch.tensor([[0., -omega], [omega, 0.]], device=device)
    for k in range(1, T):
        r = x[:, k-1, :]
        drift = (r @ A.T) - decay * r
        dw = torch.randn(B, 2, device=device) * torch.sqrt(dt) * sigma
        x[:, k, :] = x[:, k-1, :] + drift*dt + dw
    mask = torch.ones_like(x)
    return x, t, mask

def toy_double_well(B=8, T=300, sigma=0.2, device="cpu"):
    """1D double-well potential: dX = -(X^3 - X)dt + sigma dW"""
    t = torch.linspace(0, 3.0, T, device=device)
    dt = t[1] - t[0]
    x = torch.zeros(B, T, 1, device=device)
    x[:, 0, 0] = torch.randn(B, device=device) * 0.5
    for k in range(1, T):
        x_prev = x[:, k-1, 0]
        drift = -(x_prev**3 - x_prev)
        dw = torch.randn(B, device=device) * torch.sqrt(dt) * sigma
        x[:, k, 0] = x_prev + drift*dt + dw
    mask = torch.ones_like(x)
    return x, t, mask

def toy_lotka(B=8, T=400, dt=0.01, a=1.1, b=0.4, c=0.4, d=1.0, sigma=0.05, device="cpu"):
    """2D Lotka–Volterra predator–prey with small diffusion"""
    t = torch.arange(T, device=device, dtype=torch.float32) * dt
    x = torch.zeros(B, T, 2, device=device)
    x[:, 0, 0] = 1.0 + 0.1*torch.rand(B, device=device)
    x[:, 0, 1] = 1.0 + 0.1*torch.rand(B, device=device)
    for k in range(1, T):
        X, Y = x[:, k-1, 0], x[:, k-1, 1]
        dX = (a*X - b*X*Y)
        dY = (-d*Y + c*X*Y)
        dw = torch.randn(B, 2, device=device) * torch.sqrt(torch.tensor(dt, device=device)) * sigma
        x[:, k, 0] = X + dX*dt + dw[:, 0]
        x[:, k, 1] = Y + dY*dt + dw[:, 1]
        x[:, k, :] = x[:, k, :].clamp_min(1e-4)
    mask = torch.ones_like(x)
    return x, t, mask

# -------------------------------
# Irregular testers
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
    - This still satisfies 'interpolation only' since truth_time_steps == predict_time_steps.
    """
    if seed is not None:
        torch.manual_seed(seed)
    T = t.numel()
    keep = (torch.rand(T, device=t.device) < keep_prob)
    keep[0] = True                    # ensure we keep the start
    idx = torch.where(keep)[0].sort().values
    t_sub = t.index_select(0, idx)
    data_sub = data.index_select(1, idx)
    mask_sub = torch.ones_like(data_sub)
    return data_sub, t_sub, mask_sub

# -------------------------------
# Runner
# -------------------------------

def main():
    device = torch.device("cpu")  # set "cuda" if available

    # ------- choose toyset -------
    # x, t, mask = toy_ou(B=8, T=200, device=device); input_dim = 1
    # x, t, mask = toy_spiral(B=8, T=300, device=device); input_dim = 2
    # x, t, mask = toy_double_well(B=8, T=300, device=device); input_dim = 1
    x, t, mask = toy_lotka(B=8, T=400, device=device); input_dim = 2

    # ------- choose irregular test (optional) -------
    USE_MASK_IRREGULAR = True
    USE_GLOBAL_TS_SUBSAMPLE = False

    if USE_MASK_IRREGULAR:
        # per-sample missingness via mask; time grid unchanged
        x, mask = apply_mask_irregular(x, keep_prob=0.6, per_feature=False, seed=42)

    if USE_GLOBAL_TS_SUBSAMPLE:
        # global timestamp thinning; x/t/mask all sliced with the same indices
        x, t, mask = subsample_global_irregular(x, t, keep_prob=0.5, seed=123)

    # ------- build model components -------
    latent_dim = 16
    sde_func = NeuralSDEFunc(
        input_dim=input_dim,          # not directly used in current f/g but kept for API symmetry
        hidden_dim=latent_dim,
        hidden_hidden_dim=64,
        num_layers=2
    )
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
        obsrv_std=0.1
    )

    # ------- forward (interpolation only) -------
    with torch.no_grad():
        pred, info = model.get_reconstruction(
            time_steps_to_predict=t,
            data=x,
            truth_time_steps=t,
            mask=mask,
            n_traj_samples=1,
            mode="interp"
        )
    print("prediction shape =", tuple(pred.shape))  # expected: (1, B, T_kept, D)

    data_dict = {
        "observed_data": x,  # (B, T, D)
        "observed_tp": t,  # (T,)
        "observed_mask": mask,  # (B, T, D) or None
        "data_to_predict": x,  # interpolation: predict the same grid
        "tp_to_predict": t,  # (T,)
        "mask_predicted_data": mask,  # same mask for interpolation
    }

    viz = Visualizations(device)
    viz.draw_all_plots_one_dim(
        data_dict=data_dict,
        model=model,
        plot_name="toy_recon.png",
        save=True,  # saves to plots/<experimentID>/
        experimentID="toy"
    )
    print("Saved plots to: plots/toy/toy_recon.png")
if __name__ == "__main__":
    main()

# Build a data_dict in the format plotting.py expects