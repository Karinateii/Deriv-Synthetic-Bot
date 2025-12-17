"""
Market Data Handler for MT5
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable
from loguru import logger
from api.mt5_api import MT5API
import MetaTrader5 as mt5


class MT5MarketDataHandler:
    """Handles market data from MT5"""
    
    def __init__(self, mt5_api: MT5API, symbol: str):
        self.mt5_api = mt5_api
        self.symbol = symbol
        self.data = pd.DataFrame()
        self.callback = None
        
    def start(self):
        """Start collecting market data"""
        logger.info(f"Starting market data collection for {self.symbol}")
        
        # Load historical data
        self._load_historical_data()
        
        # Note: MT5 doesn't have direct tick streaming, we'll poll periodically
        logger.success(f"Market data initialized for {self.symbol}")
        
    def _load_historical_data(self):
        """Load historical bar data (more efficient than ticks)"""
        logger.info(f"Loading historical data for {self.symbol}...")
        
        # Try to get 1-minute bars first (more reliable)
        import MetaTrader5 as mt5
        df = self.mt5_api.get_rates(self.symbol, timeframe=mt5.TIMEFRAME_M1, count=500)
        
        if df.empty:
            logger.warning(f"No bar data, trying tick data for {self.symbol}")
            # Fallback to ticks
            df = self.mt5_api.get_ticks_history(self.symbol, count=5000)
            
            if df.empty:
                logger.warning(f"No historical data for {self.symbol}")
                return
                
            # Convert to OHLC format (aggregate ticks into bars)
            df = df.set_index('time')
            df = df.resample('1s').agg({  # 1-second bars
                'bid': 'ohlc',
                'ask': 'mean',
                'volume_real': 'sum'
            })
            
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() if col[1] else col[0] 
                         for col in df.columns.values]
            
            # Rename to standard format
            df = df.rename(columns={
                'bid_open': 'open',
                'bid_high': 'high',
                'bid_low': 'low',
                'bid_close': 'close',
                'ask_mean': 'ask',
                'volume_real_sum': 'volume'
            })
            
            # Drop NaN rows
            df = df.dropna()
            
            # Reset index to make time a column
            df = df.reset_index()
        else:
            # Bar data is already in correct format
            df['ask'] = df['close']  # Use close as ask for simplicity
            df['volume'] = df.get('tick_volume', df.get('real_volume', 0))
        
        self.data = df
        logger.success(f"Loaded {len(df)} bars of historical data")
        
    def update(self):
        """Update with latest bar data"""
        try:
            # Get latest 1-minute bar
            import MetaTrader5 as mt5
            df = self.mt5_api.get_rates(self.symbol, timeframe=mt5.TIMEFRAME_M1, count=2)
            
            if df.empty:
                return
                
            # Get latest bar
            latest = df.iloc[-1]
            
            # Convert to standard format
            new_bar = {
                'time': latest['time'],
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low'],
                'close': latest['close'],
                'ask': latest['close'],
                'volume': latest.get('tick_volume', latest.get('real_volume', 0))
            }
            
            # Check if this bar already exists (same timestamp)
            if len(self.data) > 0:
                last_time = self.data.iloc[-1]['time']
                if new_bar['time'] == last_time:
                    # Update existing bar
                    self.data.iloc[-1] = new_bar
                else:
                    # Add new bar
                    self.data = pd.concat([self.data, pd.DataFrame([new_bar])], ignore_index=True)
            else:
                self.data = pd.DataFrame([new_bar])
            
            # Keep only last 1000 bars
            if len(self.data) > 1000:
                self.data = self.data.tail(1000).reset_index(drop=True)
                
            # Call callback if registered
            if self.callback:
                self.callback({'tick': new_bar})
                
        except Exception as e:
            logger.error(f"Error updating data for {self.symbol}: {e}")
            
    def subscribe(self, callback: Callable):
        """Register callback for updates"""
        self.callback = callback
        
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe"""
        return self.data.copy()
