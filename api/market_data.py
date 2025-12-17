"""
Market data handler for collecting and processing real-time data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Callable
from loguru import logger
from api.deriv_api import DerivAPI


class MarketDataHandler:
    """Handles real-time market data collection and processing"""
    
    def __init__(self, api: DerivAPI, symbol: str, buffer_size: int = 5000):
        self.api = api
        self.symbol = symbol
        self.buffer_size = buffer_size
        
        # Data buffers
        self.ticks = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Callbacks
        self.tick_callbacks: list[Callable] = []
        
        # Current tick
        self.current_tick: Optional[float] = None
        self.current_time: Optional[datetime] = None
        
    def start(self):
        """Start collecting market data"""
        logger.info(f"Starting market data collection for {self.symbol}")
        
        # Load historical data first
        self._load_historical_data()
        
        # Subscribe to real-time ticks
        self.api.subscribe_ticks(self.symbol, self._on_tick)
        
    def _load_historical_data(self):
        """Load historical tick data"""
        try:
            logger.info(f"Loading historical data for {self.symbol}...")
            history = self.api.get_ticks_history(self.symbol, count=self.buffer_size)
            
            if 'times' in history and 'prices' in history:
                for timestamp, price in zip(history['times'], history['prices']):
                    self.timestamps.append(timestamp)
                    self.ticks.append(float(price))
                    
                logger.success(f"Loaded {len(self.ticks)} historical ticks")
                
                if len(self.ticks) > 0:
                    self.current_tick = self.ticks[-1]
                    self.current_time = datetime.fromtimestamp(self.timestamps[-1])
                    
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            
    def _on_tick(self, data):
        """Handle incoming tick data"""
        try:
            if 'tick' in data:
                tick_data = data['tick']
                price = float(tick_data['quote'])
                timestamp = tick_data['epoch']
                
                # Update buffers
                self.ticks.append(price)
                self.timestamps.append(timestamp)
                
                # Update current
                self.current_tick = price
                self.current_time = datetime.fromtimestamp(timestamp)
                
                # Notify callbacks
                for callback in self.tick_callbacks:
                    callback(price, timestamp)
                    
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick updates"""
        self.tick_callbacks.append(callback)
        
    def get_dataframe(self, periods: int = None) -> pd.DataFrame:
        """
        Get market data as pandas DataFrame
        
        Args:
            periods: Number of most recent periods to return (None = all)
        """
        if len(self.ticks) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': list(self.timestamps),
            'close': list(self.ticks)
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')
        
        if periods:
            df = df.tail(periods)
            
        return df
        
    def get_ohlc(self, timeframe: str = '1min', periods: int = None) -> pd.DataFrame:
        """
        Convert tick data to OHLC format
        
        Args:
            timeframe: Resampling timeframe ('1min', '5min', '15min', '1H', etc.)
            periods: Number of periods to return
        """
        df = self.get_dataframe()
        
        if df.empty:
            return pd.DataFrame()
            
        # Resample to OHLC
        ohlc = df['close'].resample(timeframe).ohlc()
        ohlc['volume'] = df['close'].resample(timeframe).count()
        
        if periods:
            ohlc = ohlc.tail(periods)
            
        return ohlc.dropna()
        
    def get_latest_ticks(self, count: int = 100) -> np.ndarray:
        """Get latest N ticks as numpy array"""
        ticks_list = list(self.ticks)
        if len(ticks_list) < count:
            return np.array(ticks_list)
        return np.array(ticks_list[-count:])
        
    def get_current_price(self) -> float:
        """Get current price"""
        return self.current_tick if self.current_tick else 0.0
        
    def get_price_change(self, periods: int = 100) -> float:
        """Get price change over N periods"""
        if len(self.ticks) < periods:
            return 0.0
            
        old_price = self.ticks[-periods]
        current_price = self.ticks[-1]
        
        return ((current_price - old_price) / old_price) * 100
        
    def get_volatility(self, periods: int = 100) -> float:
        """Calculate recent volatility (standard deviation of returns)"""
        if len(self.ticks) < periods:
            return 0.0
            
        recent_ticks = list(self.ticks)[-periods:]
        returns = np.diff(recent_ticks) / recent_ticks[:-1]
        
        return np.std(returns) * 100
        
    def is_data_ready(self, min_ticks: int = 100) -> bool:
        """Check if enough data is available"""
        return len(self.ticks) >= min_ticks
