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
    Infer y by minimizing E(x, y) via gradient descent (MAP inference).

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
        # Make y a leaf tensor so gradients are computed correctly
        y = y.clone().detach().requires_grad_(True)

        energy = model(x, y)
        energy.sum().backward()

        with torch.no_grad():
            y -= lr * y.grad

    return y.detach()


def run_continual_inference():
    # -------------------------
    # Load OLD model (trained on x ∈ [-5, 5])
    # -------------------------
    old_model = EnergyModel()
    old_model.load_state_dict(
        torch.load("ebm_model.pt", map_location="cpu")
    )

    # -------------------------
    # Load NEW model (after continual training on x ∈ [5, 8])
    # -------------------------
    new_model = EnergyModel()
    new_model.load_state_dict(
        torch.load("new_ebm_model.pt", map_location="cpu")
    )

    # -------------------------
    # Test inputs: FULL RANGE
    # -------------------------
    x_vals = np.linspace(-5.0, 8.0, 160)
    x = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)

    # -------------------------
    # MAP inference
    # -------------------------
    y_old = infer_y(old_model, x)
    y_new = infer_y(new_model, x)

    # -------------------------
    # Ground truth
    # -------------------------
    y_true = true_function(x_vals)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8, 4))

    plt.plot(
        x_vals,
        y_true,
        color="black",
        linewidth=2,
        label="True function",
    )

    plt.scatter(
        x_vals,
        y_old.numpy(),
        s=14,
        alpha=0.5,
        label="Before continual (old model)",
    )

    plt.scatter(
        x_vals,
        y_new.numpy(),
        s=14,
        alpha=0.7,
        label="After continual (new model)",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("EBM MAP Inference: Before vs After Continual Learning")
    plt.legend()
    plt.tight_layout()

    plt.savefig("ebm_continual_inference_before_vs_after.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_continual_inference()
