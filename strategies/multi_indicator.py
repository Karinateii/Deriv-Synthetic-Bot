"""
Multi-Indicator Fusion Strategy
Combines multiple technical indicators for signal generation
"""
import pandas as pd
import numpy as np
from typing import Tuple
from strategies.base_strategy import BaseStrategy
from config import Config


class MultiIndicatorStrategy(BaseStrategy):
    """Strategy combining multiple technical indicators"""
    
    def __init__(self):
        super().__init__("Multi_Indicator")
        
    def analyze(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, dict]:
        """
        Analyze using multiple technical indicators
        
        Returns:
            (signal, confidence, metadata)
        """
        if df.empty or len(df) < 50:
            return 'NEUTRAL', 0.0, {'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        metadata = {}
        
        buy_score = 0
        sell_score = 0
        max_score = 0
        
        # RSI Signal
        if 'rsi' in latest:
            max_score += 2
            if latest['rsi'] < 30:
                buy_score += 2
                metadata['rsi_signal'] = 'oversold'
            elif latest['rsi'] > 70:
                sell_score += 2
                metadata['rsi_signal'] = 'overbought'
            elif 40 <= latest['rsi'] <= 60:
                buy_score += 1
                sell_score += 1
                metadata['rsi_signal'] = 'neutral'
        
        # MACD Signal
        if 'macd' in latest and 'macd_signal' in latest:
            max_score += 2
            macd_diff = latest['macd'] - latest['macd_signal']
            
            if macd_diff > 0:
                buy_score += 2
                metadata['macd_signal'] = 'bullish'
            else:
                sell_score += 2
                metadata['macd_signal'] = 'bearish'
        
        # Bollinger Bands Signal
        if 'bb_upper' in latest and 'bb_lower' in latest:
            max_score += 2
            bb_position = (current_price - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            
            if bb_position < 0.2:
                buy_score += 2
                metadata['bb_signal'] = 'near_lower'
            elif bb_position > 0.8:
                sell_score += 2
                metadata['bb_signal'] = 'near_upper'
            
            metadata['bb_position'] = bb_position
        
        # EMA Crossover
        if 'ema_9' in latest and 'ema_21' in latest:
            max_score += 2
            
            if latest['ema_9'] > latest['ema_21']:
                buy_score += 2
                metadata['ema_signal'] = 'bullish_cross'
            else:
                sell_score += 2
                metadata['ema_signal'] = 'bearish_cross'
        
        # Stochastic
        if 'stoch_k' in latest:
            max_score += 2
            
            if latest['stoch_k'] < 20:
                buy_score += 2
                metadata['stoch_signal'] = 'oversold'
            elif latest['stoch_k'] > 80:
                sell_score += 2
                metadata['stoch_signal'] = 'overbought'
        
        # ADX Trend Strength
        if 'adx' in latest:
            max_score += 1
            
            if latest['adx'] > 25:
                # Strong trend, boost the leading signal
                if buy_score > sell_score:
                    buy_score += 1
                else:
                    sell_score += 1
                metadata['adx_signal'] = 'strong_trend'
            else:
                metadata['adx_signal'] = 'weak_trend'
        
        # Williams %R
        if 'williams_r' in latest:
            max_score += 1
            
            if latest['williams_r'] < -80:
                buy_score += 1
                metadata['williams_signal'] = 'oversold'
            elif latest['williams_r'] > -20:
                sell_score += 1
                metadata['williams_signal'] = 'overbought'
        
        # Determine signal
        metadata['buy_score'] = buy_score
        metadata['sell_score'] = sell_score
        metadata['max_score'] = max_score
        
        if buy_score > sell_score:
            signal = 'BUY'
            confidence = buy_score / max_score if max_score > 0 else 0
        elif sell_score > buy_score:
            signal = 'SELL'
            confidence = sell_score / max_score if max_score > 0 else 0
        else:
            signal = 'NEUTRAL'
            confidence = 0.5
        
        # Boost confidence if multiple indicators agree strongly
        score_diff = abs(buy_score - sell_score)
        if score_diff > max_score * 0.5:
            confidence *= 1.2
            confidence = min(1.0, confidence)
            metadata['strong_agreement'] = True
        
        return signal, confidence, metadata
