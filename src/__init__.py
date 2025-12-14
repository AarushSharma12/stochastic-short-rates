"""
Stochastic Short Rate Models.

This package provides implementations of short rate models for
interest rate modeling and fixed-income pricing.

Example:
    >>> from src import VasicekModel
    >>> model = VasicekModel(a=0.5, b=0.05, sigma=0.01)
    >>> price = model.price_bond(r_t=0.03, T=1.0)
"""

from .vasicek import VasicekModel

__all__ = ["VasicekModel"]
__version__ = "0.1.0"
