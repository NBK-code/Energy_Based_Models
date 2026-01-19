import torch
import numpy as np
import matplotlib.pyplot as plt

from model import EnergyModel
from data import true_function


def langevin_sample(
    model,
    x,
    n_steps=200,
    step_size=0.005,
    noise_scale=0.5,
):
    """
    Draw one Langevin sample for y ~ p(y | x)
    """
    model.eval()

    y = torch.zeros_like(x)

    for _ in range(n_steps):
        y = y.clone().detach().requires_grad_(True)

        energy = model(x, y)
        energy.sum().backward()

        with torch.no_grad():
            noise = torch.randn_like(y)
            y -= step_size * y.grad
            y += noise_scale * torch.sqrt(
                torch.tensor(2 * step_size)
            ) * noise

    return y.detach()


def sample_many(
    model,
    x,
    n_samples=50,
):
    """
    Draw multiple Langevin samples per x.
    """
    samples = []

    for _ in range(n_samples):
        y = langevin_sample(model, x)
        samples.append(y)

    return torch.stack(samples, dim=0)  # (S, B, 1)


def run_langevin_demo():
    # -------------------------
    # Load trained model
    # -------------------------
    model = EnergyModel()
    model.load_state_dict(
        torch.load("ebm_model.pt", map_location="cpu")
    )

    # -------------------------
    # Inputs
    # -------------------------
    x_vals = np.linspace(-7, 7, 80)
    x = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)

    # -------------------------
    # Langevin sampling
    # -------------------------
    y_samples = sample_many(
        model,
        x,
        n_samples=60,
    )  # (S, B, 1)

    y_samples = y_samples.numpy().squeeze(-1)

    # -------------------------
    # Statistics
    # -------------------------
    y_mean = y_samples.mean(axis=0)
    y_std = y_samples.std(axis=0)

    # -------------------------
    # Ground truth
    # -------------------------
    y_true = true_function(x_vals)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(7, 4))

    plt.plot(
        x_vals,
        y_true,
        color="red",
        linewidth=2,
        label="True function",
    )

    plt.plot(
        x_vals,
        y_mean,
        color="blue",
        linewidth=2,
        label="Langevin mean",
    )

    plt.fill_between(
        x_vals,
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="blue",
        alpha=0.3,
        label="±2σ uncertainty",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("EBM Langevin Inference with Uncertainty")
    plt.legend()
    plt.tight_layout()

    plt.savefig("ebm_langevin_uncertainty.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_langevin_demo()
