"""
API module initialization
"""
from .deriv_api import DerivAPI
from .market_data import MarketDataHandler

__all__ = ['DerivAPI', 'MarketDataHandler']
