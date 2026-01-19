import torch
import numpy as np
import matplotlib.pyplot as plt

from model import EnergyModel
from data import true_function


def infer_y(
    model,
    x,
    y_init=None,
    lr=0.005,
    n_steps=200,
):
    """
    Infer y by minimizing E(x, y) via gradient descent.

    Args:
        model: trained EnergyModel
        x: (batch, 1) tensor
        y_init: initial y guess (optional)
        lr: step size
        n_steps: number of gradient steps

    Returns:
        y: inferred y (detached tensor)
    """
    model.eval()

    if y_init is None:
        y = torch.zeros_like(x)
    else:
        y = y_init.clone()

    for _ in range(n_steps):
        # Make y a leaf with gradients
        y = y.clone().detach().requires_grad_(True)

        energy = model(x, y)
        energy.sum().backward()

        with torch.no_grad():
            y -= lr * y.grad

    return y.detach()


def run_inference_demo():
    # -------------------------
    # Load trained model
    # -------------------------
    model = EnergyModel()
    model.load_state_dict(
        torch.load("ebm_model.pt", map_location="cpu")
    )

    # -------------------------
    # Test inputs
    # -------------------------
    x_vals = np.linspace(-7, 7, 100)
    x = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)

    # -------------------------
    # Inference
    # -------------------------
    y_pred = infer_y(model, x)

    # -------------------------
    # Ground truth
    # -------------------------
    y_true = true_function(x_vals)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_true, color="red", linewidth=2, label="True function")
    plt.scatter(
        x_vals,
        y_pred.numpy(),
        s=15,
        color="blue",
        alpha=0.7,
        label="EBM inference",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("EBM Regression via Energy Minimization")
    plt.legend()
    plt.tight_layout()

    plt.savefig("ebm_inference.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_inference_demo()
