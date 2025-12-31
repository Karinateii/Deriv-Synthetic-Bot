"""
Regime Detection Utilities
===========================
Standalone utility functions for market regime detection
Used by RegimeAwareStrategy and can be used by other components
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from enum import Enum
from loguru import logger


class RegimeType(Enum):
    """Simplified regime types for utility functions"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME = "extreme"
    TRANSITIONING = "transitioning"


def calculate_volatility_metrics(df: pd.DataFrame, lookback: int = 100) -> Dict:
    """
    Calculate comprehensive volatility metrics
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to analyze
        
    Returns:
        Dictionary with volatility metrics
    """
    if len(df) < lookback:
        return {
            'current_volatility': 0,
            'historical_volatility': 0,
            'volatility_ratio': 1.0,
            'volatility_percentile': 50.0,
            'atr': 0,
            'is_valid': False
        }
    
    # Returns-based volatility
    returns = df['close'].pct_change().dropna()
    recent_returns = returns.tail(lookback)
    
    # Current volatility (last 10 bars)
    current_vol = recent_returns.tail(10).std() * np.sqrt(252)
    
    # Historical volatility (full lookback)
    historical_vol = recent_returns.std() * np.sqrt(252)
    
    # Volatility ratio
    vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
    
    # Volatility percentile (where current vol stands historically)
    vol_history = returns.abs().rolling(20).std().dropna()
    current_vol_measure = returns.tail(20).abs().std()
    vol_percentile = (vol_history < current_vol_measure).mean() * 100
    
    # ATR
    if 'atr' in df.columns:
        atr = df['atr'].iloc[-1]
    else:
        atr = calculate_atr(df)
    
    return {
        'current_volatility': current_vol,
        'historical_volatility': historical_vol,
        'volatility_ratio': vol_ratio,
        'volatility_percentile': vol_percentile,
        'atr': atr,
        'is_valid': True
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(df) < period + 1:
        return 0.0
    
    high = df['high'].tail(period + 1)
    low = df['low'].tail(period + 1)
    close = df['close'].tail(period + 1)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    return atr if not np.isnan(atr) else float((high - low).mean())


def calculate_hurst_exponent(prices: np.ndarray) -> float:
    """
    Calculate Hurst exponent for mean reversion analysis
    
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Args:
        prices: Array of prices
        
    Returns:
        Hurst exponent estimate
    """
    if len(prices) < 20:
        return 0.5  # Default to random walk
    
    try:
        lags = range(2, min(20, len(prices) // 4))
        tau = []
        
        for lag in lags:
            pp = np.subtract(prices[lag:], prices[:-lag])
            tau.append(np.sqrt(np.std(pp)))
        
        if len(tau) > 0 and all(t > 0 for t in tau):
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            hurst = poly[0] * 2.0
            return max(0, min(1, hurst))
        
    except Exception as e:
        logger.debug(f"Hurst calculation error: {e}")
    
    return 0.5


def calculate_trend_strength(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Calculate trend strength metrics
    
    Returns:
        Dictionary with trend metrics
    """
    if len(df) < lookback:
        return {
            'direction': 'neutral',
            'strength': 0.0,
            'r_squared': 0.0,
            'slope_normalized': 0.0
        }
    
    prices = df['close'].tail(lookback).values
    x = np.arange(len(prices))
    
    # Linear regression
    slope, intercept = np.polyfit(x, prices, 1)
    
    # Normalize slope by price
    slope_normalized = abs(slope * len(prices) / prices[-1])
    
    # R-squared
    predicted = slope * x + intercept
    ss_res = np.sum((prices - predicted) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Direction
    if slope > 0 and r_squared > 0.3:
        direction = 'up'
    elif slope < 0 and r_squared > 0.3:
        direction = 'down'
    else:
        direction = 'neutral'
    
    return {
        'direction': direction,
        'strength': slope_normalized,
        'r_squared': r_squared,
        'slope_normalized': slope_normalized
    }


def detect_regime_simple(df: pd.DataFrame) -> RegimeType:
    """
    Simple regime detection for quick checks
    
    Returns:
        RegimeType enum value
    """
    vol_metrics = calculate_volatility_metrics(df)
    
    if not vol_metrics['is_valid']:
        return RegimeType.NORMAL
    
    vol_pct = vol_metrics['volatility_percentile']
    vol_ratio = vol_metrics['volatility_ratio']
    
    # Check for transition
    if vol_ratio > 2.0:
        return RegimeType.TRANSITIONING
    
    # Check volatility levels
    if vol_pct > 90:
        return RegimeType.EXTREME
    elif vol_pct > 75:
        return RegimeType.HIGH_VOLATILITY
    elif vol_pct < 25:
        return RegimeType.LOW_VOLATILITY
    else:
        return RegimeType.NORMAL


def detect_volatility_spike(df: pd.DataFrame, threshold: float = 2.5) -> Tuple[bool, float]:
    """
    Detect if there's a volatility spike
    
    Args:
        df: Price data
        threshold: Multiple of normal volatility to trigger spike
        
    Returns:
        (is_spike, ratio)
    """
    vol_metrics = calculate_volatility_metrics(df)
    
    if not vol_metrics['is_valid']:
        return False, 1.0
    
    ratio = vol_metrics['volatility_ratio']
    return ratio > threshold, ratio


def is_market_tradeable(df: pd.DataFrame, 
                        max_vol_percentile: float = 90,
                        max_vol_ratio: float = 3.0) -> Tuple[bool, str]:
    """
    Quick check if market conditions are suitable for trading
    
    Returns:
        (is_tradeable, reason)
    """
    vol_metrics = calculate_volatility_metrics(df)
    
    if not vol_metrics['is_valid']:
        return False, "Insufficient data"
    
    if vol_metrics['volatility_percentile'] > max_vol_percentile:
        return False, f"Volatility too high: {vol_metrics['volatility_percentile']:.0f}th percentile"
    
    if vol_metrics['volatility_ratio'] > max_vol_ratio:
        return False, f"Volatility spike: {vol_metrics['volatility_ratio']:.2f}x normal"
    
    return True, "Market conditions acceptable"


def calculate_regime_based_position_size(base_risk: float,
                                         regime: RegimeType,
                                         confidence: float,
                                         vol_percentile: float) -> float:
    """
    Calculate position size multiplier based on regime
    
    Args:
        base_risk: Base risk per trade (e.g., 0.01 for 1%)
        regime: Current market regime
        confidence: Trade confidence (0-1)
        vol_percentile: Current volatility percentile
        
    Returns:
        Adjusted risk amount
    """
    # Regime multipliers
    regime_mult = {
        RegimeType.LOW_VOLATILITY: 1.2,
        RegimeType.NORMAL: 1.0,
        RegimeType.HIGH_VOLATILITY: 0.6,
        RegimeType.EXTREME: 0.3,
        RegimeType.TRANSITIONING: 0.5
    }
    
    mult = regime_mult.get(regime, 0.5)
    
    # Confidence adjustment
    conf_mult = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
    
    # Volatility adjustment
    if vol_percentile > 80:
        vol_mult = 0.6
    elif vol_percentile > 60:
        vol_mult = 0.8
    else:
        vol_mult = 1.0
    
    final_risk = base_risk * mult * conf_mult * vol_mult
    
    # Clamp to reasonable range
    min_risk = base_risk * 0.25
    max_risk = base_risk * 2.0
    
    return max(min_risk, min(max_risk, final_risk))


def find_swing_points(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Find recent swing high and low points
    
    Returns:
        Dictionary with swing levels
    """
    if len(df) < lookback:
        return {
            'swing_high': df['high'].max() if 'high' in df.columns else df['close'].max(),
            'swing_low': df['low'].min() if 'low' in df.columns else df['close'].min(),
            'swing_high_bar': 0,
            'swing_low_bar': 0
        }
    
    recent = df.tail(lookback)
    
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    
    swing_high_bar = lookback - recent['high'].values.argmax()
    swing_low_bar = lookback - recent['low'].values.argmin()
    
    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'swing_high_bar': swing_high_bar,
        'swing_low_bar': swing_low_bar
    }


def calculate_dynamic_stops(entry_price: float,
                           direction: str,
                           atr: float,
                           regime: RegimeType) -> Dict:
    """
    Calculate dynamic stop loss and take profit based on regime
    
    Returns:
        Dictionary with stop levels
    """
    # ATR multipliers based on regime
    stop_mult = {
        RegimeType.LOW_VOLATILITY: 2.0,
        RegimeType.NORMAL: 2.5,
        RegimeType.HIGH_VOLATILITY: 3.0,
        RegimeType.EXTREME: 4.0,
        RegimeType.TRANSITIONING: 3.5
    }
    
    tp_mult = {
        RegimeType.LOW_VOLATILITY: 3.0,  # Better R:R in quiet markets
        RegimeType.NORMAL: 2.5,
        RegimeType.HIGH_VOLATILITY: 2.0,  # Quick exits
        RegimeType.EXTREME: 1.5,
        RegimeType.TRANSITIONING: 2.0
    }
    
    stop_distance = atr * stop_mult.get(regime, 2.5)
    tp_distance = atr * tp_mult.get(regime, 2.5)
    
    if direction.upper() == 'BUY':
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + tp_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - tp_distance
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'stop_distance': stop_distance,
        'tp_distance': tp_distance,
        'risk_reward': tp_distance / stop_distance if stop_distance > 0 else 0
    }
