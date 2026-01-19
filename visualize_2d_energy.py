import torch
import numpy as np
import matplotlib.pyplot as plt

from data import true_function
from model import EnergyModel


def plot_2d_energy_landscape(
    model,
    x_range=(-7.0, 7.0),
    y_range=(-3.0, 3.0),
    num_points=200,
    device="cpu",
):
    model.eval()

    # -------------------------
    # Load EXACT training data
    # -------------------------
    data = torch.load("train_data.pt", map_location="cpu")
    x_data = data["x"].numpy()
    y_data = data["y"].numpy()

    # -------------------------
    # Create (x, y) grid
    # -------------------------
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    x_tensor = torch.tensor(
        X.reshape(-1, 1), dtype=torch.float32, device=device
    )
    y_tensor = torch.tensor(
        Y.reshape(-1, 1), dtype=torch.float32, device=device
    )

    # -------------------------
    # Compute energy
    # -------------------------
    with torch.no_grad():
        energy = model(x_tensor, y_tensor).cpu().numpy()

    energy = energy.reshape(num_points, num_points)

    # -------------------------
    # True function
    # -------------------------
    y_true = true_function(x)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        energy,
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Energy")

    # Training data (exact)
    plt.scatter(
        x_data,
        y_data,
        s=6,
        color="white",
        alpha=0.6,
        label="Training data",
        linewidths=0,
    )

    # True curve
    plt.plot(
        x,
        y_true,
        color="red",
        linewidth=2,
        label="True function",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Joint Energy Landscape E(x, y)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("energy_landscape_2d.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # -------------------------
    # Load trained model
    # -------------------------
    model = EnergyModel()
    model.load_state_dict(
        torch.load("ebm_model.pt", map_location="cpu")
    )

    plot_2d_energy_landscape(model)
