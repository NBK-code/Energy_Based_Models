import torch
import numpy as np
import matplotlib.pyplot as plt

from data import true_function
from model import EnergyModel


def plot_energy_landscape(
    model,
    x0=1.0,
    y_range=(-3.0, 3.0),
    num_points=400,
    device="cpu",
):
    model.eval()

    # Create y grid
    y_vals = np.linspace(y_range[0], y_range[1], num_points)
    x_vals = np.full_like(y_vals, x0)

    x = torch.tensor(x_vals, dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        energy = model(x, y).cpu().numpy()

    # True y for reference
    y_true = true_function(np.array([x0]))[0]

    plt.figure(figsize=(6, 4))
    plt.plot(y_vals, energy, label="Energy E(x, y)")
    plt.axvline(y_true, color="red", linestyle="--", label="True y")
    plt.xlabel("y")
    plt.ylabel("Energy")
    plt.title(f"Energy landscape at x = {x0:.2f}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"energy_landscape_x_{x0:.2f}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Load trained model
    model = EnergyModel()
    model.load_state_dict(torch.load("ebm_model.pt", map_location="cpu"))

    plot_energy_landscape(model, x0=1.0)
    plot_energy_landscape(model, x0=-2.0)
