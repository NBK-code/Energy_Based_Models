import torch
import numpy as np
import matplotlib.pyplot as plt

from data import true_function
from model import EnergyModel


def plot_2d_energy_landscape(
    model,
    x_range=(-7.0, 10.0),
    y_range=(-3.0, 3.0),
    num_points=250,
    device="cpu",
):
    model.eval()

    # -------------------------
    # Load OLD training data
    # -------------------------
    old_data = torch.load("train_data.pt", map_location="cpu")
    x_old = old_data["x"].numpy()
    y_old = old_data["y"].numpy()

    # -------------------------
    # Load NEW training data
    # -------------------------
    new_data = torch.load("new_train_data.pt", map_location="cpu")
    x_new = new_data["x"].numpy()
    y_new = new_data["y"].numpy()

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
    plt.figure(figsize=(8, 5))

    contour = plt.contourf(
        X,
        Y,
        energy,
        levels=60,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Energy")

    # Old training data
    plt.scatter(
        x_old,
        y_old,
        s=6,
        color="white",
        alpha=0.5,
        label="Old data (x ∈ [-5, 5])",
        linewidths=0,
    )

    # New training data
    plt.scatter(
        x_new,
        y_new,
        s=10,
        color="cyan",
        alpha=0.8,
        label="New data (x ∈ [5, 8])",
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
    plt.title("Joint Energy Landscape After Continual Learning")
    plt.legend()
    plt.tight_layout()

    plt.savefig("energy_landscape_2d_continual.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # -------------------------
    # Load continually trained model
    # -------------------------
    model = EnergyModel()
    model.load_state_dict(
        torch.load("new_ebm_model.pt", map_location="cpu")
    )

    plot_2d_energy_landscape(model)
