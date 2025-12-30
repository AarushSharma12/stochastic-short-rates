# Stochastic Short Rate Modeling Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A stochastic interest rate modeling engine that calibrates the **Vasicek model** to historical SOFR (Secured Overnight Financing Rate) data, simulates future rate paths via Monte Carlo methods, and prices fixed-income securities using both analytical and numerical techniques.

---

## 📊 Project Overview

This project implements the classic **Vasicek (1977)** short-rate model, one of the foundational models in fixed-income quantitative finance. The model describes the evolution of the instantaneous short rate as an Ornstein-Uhlenbeck process:

$$
dr_t = a(b - r_t)\,dt + \sigma\,dW_t
$$

| Parameter | Description              |
| --------- | ------------------------ |
| $a$       | Mean reversion speed     |
| $b$       | Long-term mean level     |
| $\sigma$  | Volatility               |
| $W_t$     | Standard Brownian motion |

### Why Vasicek?

- **Mean Reversion**: Captures the economic intuition that interest rates tend to revert to a long-run equilibrium.
- **Analytical Tractability**: Closed-form solutions exist for zero-coupon bond prices and European options.
- **Foundation for Advanced Models**: Understanding Vasicek is essential before moving to Hull-White, CIR, or HJM frameworks.

---

## ✨ Key Features

### 1. Model Calibration

- **Method**: OLS regression on discretized Euler-Maruyama dynamics
- **Input**: Historical time-series data (e.g., daily SOFR rates)
- **Output**: Estimated parameters $(a, b, \sigma)$

### 2. Monte Carlo Simulation

- **Method**: Euler-Maruyama discretization scheme
- **Features**: Vectorized NumPy implementation for performance
- **Output**: Simulated rate paths with configurable horizon, steps, and seed

### 3. Bond Pricing

- **Analytical**: Closed-form affine term structure formula for zero-coupon bonds
- **Monte Carlo**: Numerical estimation via path-wise discounting
- **Coupon Bonds**: Portfolio replication using zero-coupon building blocks

### 4. Yield Curve Construction

- Generate model-implied yield curves across arbitrary maturities
- Compare analytical vs. Monte Carlo pricing for validation

---

## 🏗️ Project Structure

```
stochastic-short-rates/
├── src/
│   ├── __init__.py
│   └── vasicek.py              # OOP implementation
├── notebooks/
│   └── monte-carlo.ipynb       # Research notebook with derivations
├── data/
│   └── data.xlsx               # Historical SOFR data
├── requirements.txt
└── README.md
```

### 🔧 `src/vasicek.py`

A **production-ready Python module** designed for integration into larger systems:

- **Object-Oriented Design**: Clean `VasicekModel` class encapsulating all model logic
- **Type Hints**: Full type annotations for IDE support and static analysis
- **Docstrings**: Comprehensive documentation following NumPy style
- **Input Validation**: Defensive programming with meaningful error messages
- **Vectorization**: NumPy-based implementation for computational efficiency

```python
from src.vasicek import VasicekModel

# Initialize and calibrate
model = VasicekModel(a=0.5, b=0.05, sigma=0.01)
model.calibrate(historical_rates, dt=1/252)

# Simulate 10,000 paths over 1 year
paths = model.simulate(n_paths=10_000, n_steps=252, dt=1/252, seed=42)

# Price a 5-year zero-coupon bond
price = model.price_bond(r_t=0.043, T=5.0)
```

### 📓 `monte-carlo.ipynb`

An **interactive research notebook** for exploration and validation:

- **Mathematical Derivations**: Step-by-step explanation of the Vasicek model
- **Calibration Theory**: How OLS regression connects to the SDE discretization
- **Visualizations**: Historical data plots, fan charts, yield curve comparisons
- **Model Validation**: Monte Carlo vs. analytical pricing verification

This notebook serves as both documentation and a reproducible research artifact.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/AarushSharma12/stochastic-short-rates.git
cd stochastic-short-rates

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Research Notebook

```bash
jupyter notebook notebooks/monte-carlo.ipynb
```

### Use as a Library

```python
import pandas as pd
from src.vasicek import VasicekModel

# Load your data
df = pd.read_excel("data/data.xlsx")
rates = df["Rate (%)"] / 100  # Convert to decimal

# Calibrate model
model = VasicekModel(a=1.0, b=0.05, sigma=0.01)
model.calibrate(rates, dt=1/252)

print(f"Calibrated: a={model.a:.4f}, b={model.b:.4f}, σ={model.sigma:.4f}")

# Price a bond
zcb_price = model.price_bond(r_t=model.current_rate, T=1.0)
print(f"1-Year ZCB Price: ${zcb_price:.6f}")
```

---

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
openpyxl>=3.0.0
jupyter>=1.0.0
```

---

## 📖 Mathematical Background

### Calibration via OLS Regression

The Euler discretization of the Vasicek SDE gives:

$$
r_{t+\Delta t} = r_t + a(b - r_t)\Delta t + \sigma\sqrt{\Delta t}\,Z_t
$$

Rearranging as a linear model $r_{t+\Delta t} = \alpha + \beta r_t + \varepsilon_t$:

$$
a = \frac{1 - \beta}{\Delta t}, \quad b = \frac{\alpha}{a \Delta t}, \quad \sigma = \frac{\text{std}(\varepsilon)}{\sqrt{\Delta t}}
$$

### Analytical Bond Pricing

The zero-coupon bond price $P(t, T)$ has the affine form:

$$
P(t, T) = A(t, T) \cdot e^{-B(t, T) \cdot r_t}
$$

where:

$$
B(t, T) = \frac{1 - e^{-a(T-t)}}{a}
$$

$$
A(t, T) = \exp\left[\left(b - \frac{\sigma^2}{2a^2}\right)(B - (T-t)) - \frac{\sigma^2 B^2}{4a}\right]
$$

---

## 🔬 Results Preview

| Metric                     | Value   |
| -------------------------- | ------- |
| Mean Reversion Speed ($a$) | ~0.15   |
| Long-Term Mean ($b$)       | ~4.3%   |
| Volatility ($\sigma$)      | ~0.8%   |
| 1-Year ZCB Price           | ~$0.958 |

_Values depend on calibration data range._

---

## 📚 References

1. Privault, N. (2022). _Notes on Stochastic Finance_. Chapter 17: Short Rates and Bond Pricing.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aarush Sharma**  

---
