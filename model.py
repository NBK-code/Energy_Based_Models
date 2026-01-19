import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    """
    Energy-Based Model for regression.
    Computes a scalar energy E(x, y).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        activation=nn.SiLU,
    ):
        super().__init__()

        layers = []
        in_dim = 2  # (x, y)

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # scalar energy

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """
        Small initialization is important for EBMs
        to avoid extremely steep initial energies.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1)
            y: (batch, 1)

        Returns:
            energy: (batch,)
        """
        xy = torch.cat([x, y], dim=-1)
        energy = self.net(xy)
        return energy.squeeze(-1)



if __name__ == "__main__":
    model = EnergyModel()
    x = torch.randn(4, 1)
    y = torch.randn(4, 1, requires_grad = True)

    energy = model(x, y)
    print("Energy shape:", energy.shape)

    energy.sum().backward()
    print("y grad exists:", y.grad is not None)
    print("Gradient wrt y shape:", y.grad.shape)





