"""
Vasicek Short Rate Model Implementation.

This module provides a reusable class for the Vasicek (1977) short rate model,
including calibration from historical data, Monte Carlo simulation, and
analytical zero-coupon bond pricing.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class VasicekModel:
    """
    Vasicek Short Rate Model.
    
    The Vasicek model describes the evolution of interest rates as:
        dr_t = a(b - r_t)dt + σdW_t
    
    Where:
        - a: Mean reversion speed
        - b: Long-term mean rate
        - σ: Volatility
        - W_t: Standard Brownian motion
    
    Attributes:
        a (float): Mean reversion speed parameter.
        b (float): Long-term mean rate parameter.
        sigma (float): Volatility parameter.
        current_rate (float | None): Current short rate level.
    
    Example:
        >>> model = VasicekModel(a=0.5, b=0.05, sigma=0.01, current_rate=0.03)
        >>> paths = model.simulate(n_paths=1000, n_steps=252, dt=1/252)
        >>> bond_price = model.price_bond(r_t=0.03, T=1.0)
    """
    
    def __init__(
        self,
        a: float,
        b: float,
        sigma: float,
        current_rate: Optional[float] = None
    ) -> None:
        """
        Initialize the Vasicek model with given parameters.
        
        Args:
            a: Mean reversion speed. Must be positive for stationarity.
            b: Long-term mean rate level.
            sigma: Volatility of the short rate. Must be non-negative.
            current_rate: Current short rate level. If None, defaults to b.
        
        Raises:
            ValueError: If a <= 0 or sigma < 0.
        """
        if a <= 0:
            raise ValueError(f"Mean reversion speed 'a' must be positive, got {a}")
        if sigma < 0:
            raise ValueError(f"Volatility 'sigma' must be non-negative, got {sigma}")
        
        self.a = a
        self.b = b
        self.sigma = sigma
        self.current_rate = current_rate if current_rate is not None else b
    
    def calibrate(self, data: pd.Series, dt: float = 1/252) -> "VasicekModel":
        """
        Calibrate model parameters from historical rate data using OLS regression.
        
        Uses the discretized Vasicek dynamics:
            r_{t+1} - r_t = a(b - r_t)Δt + σε√Δt
        
        Which can be rewritten as a linear regression:
            Δr_t = α + β*r_t + ε
        
        Where:
            - β = -a*Δt  →  a = -β/Δt
            - α = a*b*Δt  →  b = α/(a*Δt) = -α/β
            - σ = std(residuals) / √Δt
        
        Args:
            data: Time series of historical short rates (e.g., daily rates).
            dt: Time step between observations (default: 1/252 for daily data).
        
        Returns:
            Self, with updated parameters a, b, sigma, and current_rate.
        
        Raises:
            ValueError: If data has fewer than 3 observations.
        """
        if len(data) < 3:
            raise ValueError("Need at least 3 observations for calibration")
        
        # Compute rate changes
        r_t = data.values[:-1]
        r_next = data.values[1:]
        delta_r = r_next - r_t
        
        # OLS regression: Δr = α + β*r_t + ε
        # Using normal equations: [α, β] = (X'X)^(-1) X'y
        X = np.column_stack([np.ones_like(r_t), r_t])
        y = delta_r
        
        # Solve using least squares
        coefficients, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = coefficients
        
        # Extract Vasicek parameters
        self.a = -beta / dt
        self.b = alpha / (self.a * dt) if self.a != 0 else data.mean()
        
        # Estimate volatility from residuals
        fitted = X @ coefficients
        residual_errors = y - fitted
        self.sigma = np.std(residual_errors, ddof=2) / np.sqrt(dt)
        
        # Set current rate to last observation
        self.current_rate = data.iloc[-1]
        
        return self
    
    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate short rate paths using the Euler-Maruyama method.
        
        The discretization scheme is:
            r_{t+Δt} = r_t + a(b - r_t)Δt + σ√Δt * Z
        
        Where Z ~ N(0, 1).
        
        Args:
            n_paths: Number of Monte Carlo paths to simulate.
            n_steps: Number of time steps per path.
            dt: Time step size (e.g., 1/252 for daily steps).
            seed: Random seed for reproducibility.
        
        Returns:
            np.ndarray: Array of shape (n_paths, n_steps + 1) containing
                simulated rate paths. Column 0 is the initial rate.
        
        Example:
            >>> model = VasicekModel(a=0.5, b=0.05, sigma=0.01, current_rate=0.03)
            >>> paths = model.simulate(n_paths=10000, n_steps=252, dt=1/252, seed=42)
            >>> paths.shape
            (10000, 253)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-allocate paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.current_rate
        
        # Pre-generate all random shocks for efficiency
        dW = np.random.standard_normal((n_paths, n_steps))
        
        # Euler-Maruyama iteration (vectorized over paths)
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            r_t = paths[:, t]
            drift = self.a * (self.b - r_t) * dt
            diffusion = self.sigma * sqrt_dt * dW[:, t]
            paths[:, t + 1] = r_t + drift + diffusion
        
        return paths
    
    def price_bond(
        self,
        r_t: float,
        T: float,
        t: float = 0
    ) -> float:
        """
        Calculate the analytical zero-coupon bond price using the affine term structure.
        
        For the Vasicek model, the ZCB price P(t, T) has the closed-form solution:
            P(t, T) = A(t, T) * exp(-B(t, T) * r_t)
        
        Where:
            B(t, T) = (1 - exp(-a(T-t))) / a
            A(t, T) = exp[(B(t,T) - T + t)(a²b - σ²/2) / a² - σ²B(t,T)² / (4a)]
        
        Args:
            r_t: Current short rate level.
            T: Maturity time of the bond.
            t: Current time (default: 0).
        
        Returns:
            float: Zero-coupon bond price P(t, T).
        
        Raises:
            ValueError: If T < t (maturity before current time).
        
        Example:
            >>> model = VasicekModel(a=0.5, b=0.05, sigma=0.01)
            >>> price = model.price_bond(r_t=0.03, T=1.0, t=0)
            >>> print(f"1-year ZCB price: {price:.6f}")
        """
        if T < t:
            raise ValueError(f"Maturity T ({T}) must be >= current time t ({t})")
        
        if T == t:
            return 1.0
        
        tau = T - t  # Time to maturity
        
        # Calculate B(t, T)
        B = (1 - np.exp(-self.a * tau)) / self.a
        
        # Calculate A(t, T)
        # A(t,T) = exp[(B - τ)(a²b - σ²/2)/a² - σ²B²/(4a)]
        a_sq = self.a ** 2
        sigma_sq = self.sigma ** 2
        
        term1 = (B - tau) * (a_sq * self.b - sigma_sq / 2) / a_sq
        term2 = sigma_sq * B ** 2 / (4 * self.a)
        A = np.exp(term1 - term2)
        
        # Bond price
        price = A * np.exp(-B * r_t)
        
        return price
    
    def yield_curve(
        self,
        r_t: float,
        maturities: np.ndarray,
        t: float = 0
    ) -> np.ndarray:
        """
        Calculate the yield curve for given maturities.
        
        The continuously compounded yield is:
            y(t, T) = -ln(P(t, T)) / (T - t)
        
        Args:
            r_t: Current short rate level.
            maturities: Array of maturity times.
            t: Current time (default: 0).
        
        Returns:
            np.ndarray: Array of yields corresponding to each maturity.
        """
        yields = np.zeros_like(maturities, dtype=float)
        
        for i, T in enumerate(maturities):
            if T > t:
                price = self.price_bond(r_t, T, t)
                yields[i] = -np.log(price) / (T - t)
            else:
                yields[i] = r_t  # Instantaneous rate
        
        return yields
    
    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"VasicekModel(a={self.a:.6f}, b={self.b:.6f}, "
            f"sigma={self.sigma:.6f}, current_rate={self.current_rate:.6f})"
        )
    
    @property
    def long_term_variance(self) -> float:
        """
        Calculate the long-term (stationary) variance of the short rate.
        
        For the Vasicek model: Var(r_∞) = σ² / (2a)
        """
        return self.sigma ** 2 / (2 * self.a)
    
    @property
    def long_term_std(self) -> float:
        """Calculate the long-term standard deviation of the short rate."""
        return np.sqrt(self.long_term_variance)
