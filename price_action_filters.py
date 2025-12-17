"""
Trade quality filters for price action strategy
- Trend strength detection
- Volatility spike detection
"""

import numpy as np
import pandas as pd


def calculate_trend_strength(data: pd.DataFrame) -> float:
    """
    Calculate trend strength (similar to ADX but simpler)
    Returns value between 0-1 where:
    - 0.0-0.3: Weak/choppy
    - 0.3-0.6: Moderate trend
    - 0.6-1.0: Strong trend
    """
    if len(data) < 50:
        return 0.0
    
    closes = data['close'].values[-50:]
    highs = data['high'].values[-50:]
    lows = data['low'].values[-50:]
    
    # Method 1: Directional movement consistency
    moves = np.diff(closes)
    positive_moves = moves[moves > 0]
    negative_moves = moves[moves < 0]
    
    if len(moves) == 0:
        return 0.0
    
    # Check if moves are consistent in one direction
    directional_consistency = abs(len(positive_moves) - len(negative_moves)) / len(moves)
    
    # Method 2: Higher highs / lower lows pattern
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]
    earlier_highs = highs[-40:-20]
    earlier_lows = lows[-40:-20]
    
    higher_highs = recent_highs.max() > earlier_highs.max()
    higher_lows = recent_lows.min() > earlier_lows.min()
    lower_highs = recent_highs.max() < earlier_highs.max()
    lower_lows = recent_lows.min() < earlier_lows.min()
    
    # Strong uptrend or downtrend pattern
    pattern_strength = 0.0
    if higher_highs and higher_lows:
        pattern_strength = 0.4
    elif lower_highs and lower_lows:
        pattern_strength = 0.4
    
    # Method 3: Price range progression
    # Strong trends show expanding ranges
    recent_range = recent_highs.max() - recent_lows.min()
    earlier_range = earlier_highs.max() - earlier_lows.min()
    
    if earlier_range > 0:
        range_expansion = min(recent_range / earlier_range, 2.0) / 2.0
    else:
        range_expansion = 0.0
    
    # Combine methods (weighted average)
    trend_strength = (
        directional_consistency * 0.4 +
        pattern_strength +
        range_expansion * 0.2
    )
    
    return min(trend_strength, 1.0)


def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(data) < period + 1:
        return 0.0
    
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    tr_list = []
    for i in range(1, len(data)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0
    
    return np.mean(tr_list[-period:])


def is_volatility_spike(data: pd.DataFrame, threshold: float = 2.5) -> bool:
    """
    Detect if current volatility is abnormally high
    Returns True if current ATR > threshold * average ATR
    """
    if len(data) < 50:
        return False
    
    # Calculate current ATR (last 14 bars)
    current_atr = calculate_atr(data.tail(20), period=14)
    
    # Calculate average ATR over longer period
    avg_atr = calculate_atr(data.tail(50), period=14)
    
    if avg_atr == 0:
        return False
    
    volatility_ratio = current_atr / avg_atr
    
    return volatility_ratio > threshold
