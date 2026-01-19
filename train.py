import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data import generate_data
from model import EnergyModel


def sample_negative_y(
    y_pos: torch.Tensor,
    y_range=(-3.0, 3.0)
):
    """
    Uniform negative sampling for EBMs.
    """
    y_neg = torch.empty_like(y_pos)
    y_neg.uniform_(y_range[0], y_range[1])
    return y_neg


def train(
    n_epochs=300,
    batch_size=128,
    lr=1e-3,
    device="cpu",
):
    # -------------------------
    # Data
    # -------------------------

    x_np, y_np = generate_data(n=2000)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    torch.save({"x": x, "y": y}, "train_data.pt")


    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Model
    # -------------------------
    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(n_epochs):
        total_loss = 0.0

        for x_batch, y_pos in loader:
            x_batch = x_batch.to(device)
            y_pos = y_pos.to(device)

            # Negative samples
            y_neg = sample_negative_y(y_pos)

            # Energies
            energy_pos = model(x_batch, y_pos)
            energy_neg = model(x_batch, y_neg)

            # Contrastive loss (stable)
            loss = F.softplus(energy_pos - energy_neg).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ebm_model.pt")

    return model


if __name__ == "__main__":
    train()
