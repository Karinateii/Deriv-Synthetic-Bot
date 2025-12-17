"""
Technical Indicators Calculator
"""
import pandas as pd
import numpy as np
from typing import Tuple
import talib as ta


class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty or len(df) < 50:
            return df
            
        df = df.copy()
        
        try:
            # Moving Averages
            df['ema_9'] = ta.EMA(df['close'], timeperiod=9)
            df['ema_21'] = ta.EMA(df['close'], timeperiod=21)
            df['ema_50'] = ta.EMA(df['close'], timeperiod=50)
            df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
            
            # RSI
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
                df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
                df['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = ta.STOCH(
                df['high'], 
                df['low'], 
                df['close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            
            # ATR (Average True Range)
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # ADX (Average Directional Index)
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # Williams %R
            df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # OBV (On Balance Volume)
            if 'volume' in df.columns:
                df['obv'] = ta.OBV(df['close'], df['volume'])
            
            # ROC (Rate of Change)
            df['roc'] = ta.ROC(df['close'], timeperiod=10)
            
            # MOM (Momentum)
            df['momentum'] = ta.MOM(df['close'], timeperiod=10)
            
            # Custom indicators
            df = TechnicalIndicators._add_custom_indicators(df)
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            
        return df
    
    @staticmethod
    def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators"""
        
        # Price position relative to Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
            df['bb_width'] = bb_range / df['bb_middle']
        
        # Trend strength
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            df['trend_strength'] = (df['ema_9'] - df['ema_21']) / df['ema_21'] * 100
        
        # Volatility ratio
        if 'atr' in df.columns:
            df['volatility_ratio'] = df['atr'] / df['close'] * 100
        
        # Price momentum
        df['price_change'] = df['close'].pct_change() * 100
        df['price_change_5'] = df['close'].pct_change(5) * 100
        df['price_change_10'] = df['close'].pct_change(10) * 100
        
        # Support/Resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        return df
    
    @staticmethod
    def detect_trend(df: pd.DataFrame) -> str:
        """
        Detect current trend
        
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        if df.empty or len(df) < 50:
            return 'sideways'
            
        latest = df.iloc[-1]
        
        # Check EMA alignment
        if 'ema_9' in latest and 'ema_21' in latest and 'ema_50' in latest:
            if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
                return 'uptrend'
            elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
                return 'downtrend'
        
        # Check ADX strength
        if 'adx' in latest:
            if latest['adx'] < 25:
                return 'sideways'
        
        return 'sideways'
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect RSI divergence
        
        Returns:
            (has_divergence, divergence_type)
        """
        if df.empty or len(df) < 50:
            return False, 'none'
            
        # Get recent data
        recent = df.tail(20)
        
        if 'rsi' not in recent.columns:
            return False, 'none'
        
        # Find price peaks and troughs
        price_peaks = recent['close'].iloc[-10:].nlargest(2)
        price_troughs = recent['close'].iloc[-10:].nsmallest(2)
        
        # Find RSI peaks and troughs
        rsi_peaks = recent['rsi'].iloc[-10:].nlargest(2)
        rsi_troughs = recent['rsi'].iloc[-10:].nsmallest(2)
        
        # Bullish divergence: price lower low, RSI higher low
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if (price_troughs.iloc[0] < price_troughs.iloc[1] and 
                rsi_troughs.iloc[0] > rsi_troughs.iloc[1]):
                return True, 'bullish'
        
        # Bearish divergence: price higher high, RSI lower high
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks.iloc[0] > price_peaks.iloc[1] and 
                rsi_peaks.iloc[0] < rsi_peaks.iloc[1]):
                return True, 'bearish'
        
        return False, 'none'
    
    @staticmethod
    def get_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        """
        Calculate support and resistance levels
        
        Returns:
            (support_level, resistance_level)
        """
        if df.empty or len(df) < lookback:
            return 0.0, 0.0
            
        recent = df.tail(lookback)
        support = recent['low'].min()
        resistance = recent['high'].max()
        
        return support, resistance
    
    @staticmethod
    def calculate_signal_strength(df: pd.DataFrame) -> float:
        """
        Calculate overall signal strength (0-100)
        
        Combines multiple indicators to determine signal quality
        """
        if df.empty or len(df) < 50:
            return 0.0
            
        latest = df.iloc[-1]
        score = 0
        max_score = 0
        
        # RSI signals
        if 'rsi' in latest:
            max_score += 20
            if latest['rsi'] < 30:
                score += 20  # Oversold
            elif latest['rsi'] > 70:
                score += 20  # Overbought
            elif 30 <= latest['rsi'] <= 70:
                score += 10  # Neutral
        
        # MACD signals
        if 'macd' in latest and 'macd_signal' in latest:
            max_score += 20
            if latest['macd'] > latest['macd_signal']:
                score += 20  # Bullish
            else:
                score += 10
        
        # Bollinger Band signals
        if 'bb_position' in latest:
            max_score += 20
            if latest['bb_position'] < 0.2 or latest['bb_position'] > 0.8:
                score += 20  # Near bands
            else:
                score += 10
        
        # Trend strength
        if 'adx' in latest:
            max_score += 20
            if latest['adx'] > 25:
                score += 20  # Strong trend
            else:
                score += 10
        
        # Stochastic
        if 'stoch_k' in latest:
            max_score += 20
            if latest['stoch_k'] < 20 or latest['stoch_k'] > 80:
                score += 20
            else:
                score += 10
        
        if max_score == 0:
            return 0.0
            
        return (score / max_score) * 100


class SimpleIndicators:
    """Simple indicators for when TA-Lib is not available"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
