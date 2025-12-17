"""
Strategies module initialization
"""
from .base_strategy import BaseStrategy
from .ml_strategy import MLStrategy
from .multi_indicator import MultiIndicatorStrategy

__all__ = ['BaseStrategy', 'MLStrategy', 'MultiIndicatorStrategy']
