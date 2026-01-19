# Energy-Based Model for Regression

This project implements a **small but complete Energy-Based Model (EBM)** for a regression problem.  
Unlike standard regression models that directly predict outputs, an EBM **learns an energy function over joint (x, y) space** and performs inference via **optimization and sampling**.

The project demonstrates:

- Energy-based modeling for regression
- Contrastive training with negative samples
- MAP inference via energy minimization
- Langevin dynamics for probabilistic inference

---

## 1. Problem Setup

We consider a regression problem with scalar input `x` and scalar output `y`.

The data is generated from the noisy function:

![y = sin(x) + 0.1x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)](https://latex.codecogs.com/svg.latex?y%3D%5Csin(x)%2B0.1x%2B%5Cepsilon%2C%5Cquad%5Cepsilon%5Csim%5Cmathcal%7BN%7D(0%2C%5Csigma%5E2))

This dataset is chosen because it is:

- Nonlinear but smooth
- Unimodal for each fixed `x`
- Easy to visualize in joint `(x, y)` space
- Suitable for illustrating energy landscapes

---

## 2. Energy-Based Model

Instead of learning a predictor `y = f(x)`, we learn an **energy function**:

![E_\theta(x, y) : \mathbb{R}^2 \rightarrow \mathbb{R}](https://latex.codecogs.com/svg.latex?E_%5Ctheta(x%2Cy)%20%3A%20%5Cmathbb%7BR%7D%5E2%20%5Crightarrow%20%5Cmathbb%7BR%7D)

The energy function assigns **low values** to compatible `(x, y)` pairs and **high values** to implausible ones.

This defines an implicit conditional distribution:

![p_\theta(y \mid x) \propto \exp(-E_\theta(x,y))](https://latex.codecogs.com/svg.latex?p_%7B%5Ctheta%7D(y%20%5Cmid%20x)%20%5Cpropto%20%5Cexp(-E_%7B%5Ctheta%7D(x%2Cy)))

No explicit likelihood or normalization constant is required.

---

## 3. Training Objective

Training is performed using **contrastive learning**.

For each input `x`:
- `y⁺` is the true target (positive sample)
- `y⁻` is a randomly sampled negative target

The loss function is:

![\mathcal{L}(x) = \operatorname{softplus}(E_\theta(x, y^+) - E_\theta(x, y^-))](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D(x)%3D%5Coperatorname%7Bsoftplus%7D(E_%5Ctheta(x%2Cy%5E%2B)-E_%5Ctheta(x%2Cy%5E-)))

where

![\operatorname{softplus}(z) = \log(1 + e^z)](https://latex.codecogs.com/svg.latex?%5Coperatorname%7Bsoftplus%7D(z)%3D%5Clog(1%2Be%5Ez))

This encourages the energy ordering:

![E_\theta(x, y^+) < E_\theta(x, y^-)](https://latex.codecogs.com/svg.latex?E_%5Ctheta(x%2Cy%5E%2B)%20%3C%20E_%5Ctheta(x%2Cy%5E-))

---

## 4. Energy Geometry

After training, the learned energy function forms a **low-energy valley** in joint `(x, y)` space.

- The valley floor follows the true regression curve
- Energy increases smoothly away from the data manifold
- The width of the valley reflects noise and uncertainty

This geometric interpretation is central to energy-based models.

---

## 5. MAP Inference (Deterministic Prediction)

Given a fixed input `x`, prediction is performed by minimizing energy:

![\hat{y}(x) = \arg\min_y E_\theta(x, y)](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D(x)%3D%5Carg%5Cmin_y%20E_%5Ctheta(x%2Cy))

This is implemented using gradient descent:

![y_{k+1} = y_k - r \nabla_y E_\theta(x, y_k)](https://latex.codecogs.com/svg.latex?y_%7Bk%2B1%7D%3Dy_k-r%5Cnabla_yE_%5Ctheta(x%2Cy_k))

This procedure corresponds to **MAP inference** (maximum a posteriori).

---

## 6. Langevin Inference (Probabilistic Prediction)

To sample from the full conditional distribution, we use **Langevin dynamics**:

![y_{k+1} = y_k - r \nabla_y E_\theta(x, y_k) + \sqrt{2r}\,\xi_k,\quad \xi_k \sim \mathcal{N}(0,1)](https://latex.codecogs.com/svg.latex?y_%7Bk%2B1%7D%3Dy_k-r%5Cnabla_yE_%5Ctheta(x%2Cy_k)%2B%5Csqrt%7B2r%7D%5Cxi_k%2C%5Cquad%5Cxi_k%5Csim%5Cmathcal%7BN%7D(0%2C1))

This update combines:
- Gradient-driven movement toward low energy
- Stochastic noise for exploration

The stationary distribution approximates:

![p_\theta(y \mid x)](https://latex.codecogs.com/svg.latex?p_%7B%5Ctheta%7D(y%20%5Cmid%20x))

---
