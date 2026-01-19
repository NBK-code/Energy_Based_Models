import numpy as np

def generate_data(
    n: int = 1000,
    x_range: tuple = (-5.0, 5.0),
    noise_std: float = 0.1,
    seed: int = 0,
):
    """
    Generate data for EBM regression.

    y = sin(x) + 0.1 * x + ε

    Args:
        n: Number of samples
        x_range: (min_x, max_x)
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        x: (n, 1) numpy array
        y: (n, 1) numpy array
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(x_range[0], x_range[1], size=(n, 1))
    noise = noise_std * rng.normal(size=(n, 1))
    y = np.sin(x) + 0.1 * x + noise

    return x.astype(np.float32), y.astype(np.float32)


def generate_grid(
    x_range=(-5.0, 5.0),
    y_range=(-3.0, 3.0),
    num_points=200,
):
    """
    Generate a grid over (x, y) space.
    Useful for visualizing energy landscapes.

    Returns:
        X: (num_points, num_points)
        Y: (num_points, num_points)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


def true_function(x):
    """
    Ground-truth noiseless function.
    Useful for evaluation and plotting.
    """
    return np.sin(x) + 0.1 * x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y = generate_data()
    plt.scatter(x, y, s=10, alpha=0.5, label="data")

    x_line = np.linspace(-5, 5, 500)
    y_line = true_function(x_line)

    plt.plot(x_line, y_line, color="red", linewidth=2, label="true function")
    plt.legend()
    plt.savefig("data_preview.png", dpi=150)
    plt.close()


