"""
Base Strategy Class
All trading strategies inherit from this
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional
from loguru import logger


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_signal = 'NEUTRAL'
        self.last_confidence = 0.0
        
    @abstractmethod
    def analyze(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, dict]:
        """
        Analyze market data and generate signal
        
        Args:
            df: DataFrame with OHLC data and indicators
            current_price: Current market price
            
        Returns:
            (signal, confidence, metadata)
            - signal: 'BUY', 'SELL', or 'NEUTRAL'
            - confidence: 0.0 to 1.0
            - metadata: Additional information about the signal
        """
        pass
    
    def get_signal(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, dict]:
        """
        Get trading signal with validation
        
        Returns:
            (signal, confidence, metadata)
        """
        try:
            signal, confidence, metadata = self.analyze(df, current_price)
            
            # Validate signal
            if signal not in ['BUY', 'SELL', 'NEUTRAL']:
                logger.warning(f"{self.name}: Invalid signal {signal}, defaulting to NEUTRAL")
                signal = 'NEUTRAL'
                confidence = 0.0
            
            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"{self.name}: Invalid confidence {confidence}, clamping")
                confidence = max(0.0, min(1.0, confidence))
            
            self.last_signal = signal
            self.last_confidence = confidence
            
            return signal, confidence, metadata
            
        except Exception as e:
            logger.error(f"{self.name} analysis error: {e}")
            return 'NEUTRAL', 0.0, {'error': str(e)}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
