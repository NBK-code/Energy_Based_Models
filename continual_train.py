import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from new_data import generate_data
from model import EnergyModel


def sample_negative_y(
    y_pos: torch.Tensor,
    y_range=(-3.0, 3.0),
):
    """
    Uniform negative sampling for EBMs.
    """
    y_neg = torch.empty_like(y_pos)
    y_neg.uniform_(y_range[0], y_range[1])
    return y_neg


def continual_train(
    n_epochs=300,
    batch_size=128,
    lr=1e-3,
    device="cpu",
):
    # -------------------------
    # New data only (x in [5, 8])
    # -------------------------
    x_np, y_np = generate_data(n=300)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    torch.save({"x": x, "y": y}, "new_train_data.pt")

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Load existing model
    # -------------------------
    model = EnergyModel().to(device)
    model.load_state_dict(
        torch.load("ebm_model.pt", map_location=device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Continual training loop
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

            # Contrastive loss
            loss = F.softplus(energy_pos - energy_neg).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"[Continual] Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

    # -------------------------
    # Save updated model
    # -------------------------
    torch.save(model.state_dict(), "new_ebm_model.pt")

    return model


if __name__ == "__main__":
    continual_train()
