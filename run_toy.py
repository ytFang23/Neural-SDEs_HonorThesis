import torch

# project imports
from lib.de_func import NeuralSDEFunc
from lib.diffeq_solver import SDESolver
from lib.sde_rnn import SDE_RNN
from lib.plotting import Visualizations
import matplotlib
from generate_timeseries import Periodic_1d
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
    # ------------- setup -------------
    torch.manual_seed(42)
    device = torch.device("cpu")  # change to "cuda" if available

    # ------------- choose toyset -------------
    # x: (B,T,D), t: (T,), mask: (B,T,D)
    # x, t, mask = toy_ou(B=8, T=200, device=device);        input_dim = 1
    #x, t, mask = toy_spiral(B=8, T=300, device=device);      input_dim = 2
    # x, t, mask = toy_double_well(B=8, T=300, device=device); input_dim = 1
    x, t, mask = toy_lotka(B=8, T=400, device=device);     input_dim = 2

    # ------------- optional irregular sampling -------------
    USE_MASK_IRREGULAR = False
    USE_GLOBAL_TS_SUBSAMPLE = True

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
        obsrv_std=0.01  # smaller noise helps gradients on toy data
    ).to(device)

    # ------------- quick training loop (interpolation) -------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 200
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
# Build a data_dict in the format plotting.py expects