"""
Pure Price Action Trading Strategy
- Support/Resistance levels
- Candlestick patterns (Pin bars, Engulfing, Inside bars)
- Multi-timeframe analysis
- Market structure (Higher highs, Lower lows, Trend breaks)
- Volume confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger


class PriceActionStrategy:
    """
    Pure price action strategy based on:
    1. Support/Resistance levels (pivot points)
    2. Candlestick patterns
    3. Market structure (trends, breakouts)
    4. Volume analysis
    5. Multi-timeframe confirmation
    """
    
    def __init__(self):
        self.name = "price_action"
        self.lookback = 100  # Bars to analyze
        self.sr_sensitivity = 0.0015  # 0.15% for S/R levels
        
        # Multi-timeframe settings
        self.timeframes = {
            'short': 50,    # ~50 ticks (micro structure)
            'medium': 100,  # ~100 ticks (entry timeframe)
            'long': 200     # ~200 ticks (trend context)
        }
        
        # Store structure levels for stop/TP placement
        self.last_analysis = {
            'support_levels': [],
            'resistance_levels': [],
            'swing_high': None,
            'swing_low': None,
            'entry_reason': None
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict:
        """Analyze price action and generate signals with multi-timeframe confirmation"""
        if len(data) < self.timeframes['long']:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        # STEP 1: Get higher timeframe bias (long-term trend)
        htf_bias = self._get_higher_timeframe_bias(data)
        
        signals = []
        reasons = []
        
        # Get current candle and recent data
        recent = data.tail(self.lookback).copy()
        current = recent.iloc[-1]
        prev = recent.iloc[-2]
        
        # 1. SUPPORT/RESISTANCE ANALYSIS
        sr_signal = self._analyze_support_resistance(recent, current)
        if sr_signal:
            signals.append(sr_signal['signal'])
            reasons.append(sr_signal['reason'])
        
        # 2. CANDLESTICK PATTERNS
        candle_signal = self._analyze_candlestick_patterns(recent)
        if candle_signal:
            signals.append(candle_signal['signal'])
            reasons.append(candle_signal['reason'])
        
        # 3. MARKET STRUCTURE (Trend)
        structure_signal = self._analyze_market_structure(recent)
        if structure_signal:
            signals.append(structure_signal['signal'])
            reasons.append(structure_signal['reason'])
        
        # 4. VOLUME CONFIRMATION
        volume_signal = self._analyze_volume(recent)
        if volume_signal:
            signals.append(volume_signal['signal'])
            reasons.append(volume_signal['reason'])
        
        # 5. PRICE MOMENTUM
        momentum_signal = self._analyze_momentum(recent)
        if momentum_signal:
            signals.append(momentum_signal['signal'])
            reasons.append(momentum_signal['reason'])
        
        # Calculate final signal
        if not signals:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'No clear setup'
            }
        
        # Count votes
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        total_votes = len(signals)
        
        if buy_votes > sell_votes:
            signal = 'BUY'
            confidence = (buy_votes / total_votes) * 100
        elif sell_votes > buy_votes:
            signal = 'SELL'
            confidence = (sell_votes / total_votes) * 100
        else:
            signal = 'NEUTRAL'
            confidence = 50.0
        
        # STEP 2: Multi-timeframe filter
        # Only take trades aligned with higher timeframe bias
        if htf_bias and htf_bias != signal and signal != 'NEUTRAL':
            # Signal against higher timeframe trend - reduce confidence or filter out
            if confidence < 80:  # Only take counter-trend if very strong
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'reason': f'Filtered: {signal} signal against {htf_bias} HTF trend'
                }
            else:
                confidence *= 0.7  # Reduce confidence for counter-trend trades
                reasons.append(f'Counter-trend to {htf_bias} HTF')
        elif htf_bias and htf_bias == signal:
            # Signal aligned with higher timeframe - boost confidence
            confidence = min(confidence * 1.2, 100)
            reasons.append(f'{htf_bias} HTF aligned')
        
        # STEP 3: Calculate structure-based stop loss and take profit
        stop_loss, take_profit = self._calculate_structure_levels(recent, signal)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': ' | '.join(reasons[:3]),  # Top 3 reasons
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _get_higher_timeframe_bias(self, data: pd.DataFrame) -> Optional[str]:
        """
        Determine overall trend bias from higher timeframe (last 200 ticks)
        This acts as a filter - only take trades aligned with HTF trend
        """
        if len(data) < self.timeframes['long']:
            return None
        
        htf_data = data.tail(self.timeframes['long'])
        
        # Calculate trend using multiple methods
        closes = htf_data['close'].values
        highs = htf_data['high'].values
        lows = htf_data['low'].values
        
        # Method 1: Simple moving average slope
        first_half_avg = closes[:100].mean()
        second_half_avg = closes[100:].mean()
        
        # Method 2: Higher highs / Lower lows
        first_half_high = highs[:100].max()
        second_half_high = highs[100:].max()
        first_half_low = lows[:100].min()
        second_half_low = lows[100:].min()
        
        # Method 3: Price position (where is current price vs HTF range)
        htf_high = highs.max()
        htf_low = lows.min()
        htf_range = htf_high - htf_low
        current_price = closes[-1]
        
        if htf_range == 0:
            return None
        
        price_position = (current_price - htf_low) / htf_range  # 0-1 scale
        
        # Voting system for HTF bias
        votes = []
        
        # Vote 1: MA slope
        if second_half_avg > first_half_avg * 1.005:  # 0.5% higher
            votes.append('BUY')
        elif second_half_avg < first_half_avg * 0.995:  # 0.5% lower
            votes.append('SELL')
        
        # Vote 2: Higher highs/lows
        if second_half_high > first_half_high and second_half_low > first_half_low:
            votes.append('BUY')
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            votes.append('SELL')
        
        # Vote 3: Price position in range
        if price_position > 0.6:  # Upper 40% of range
            votes.append('BUY')
        elif price_position < 0.4:  # Lower 40% of range
            votes.append('SELL')
        
        # Vote 4: Recent momentum (last 50 bars)
        recent_start = closes[-50]
        recent_end = closes[-1]
        momentum = (recent_end - recent_start) / recent_start
        
        if momentum > 0.01:  # 1% up
            votes.append('BUY')
        elif momentum < -0.01:  # 1% down
            votes.append('SELL')
        
        # Determine HTF bias
        if not votes:
            return None
        
        buy_votes = votes.count('BUY')
        sell_votes = votes.count('SELL')
        
        # Need strong consensus (at least 3 out of 4 votes)
        if buy_votes >= 3:
            return 'BUY'
        elif sell_votes >= 3:
            return 'SELL'
        
        return None  # No clear HTF trend
    
    def _analyze_support_resistance(self, data: pd.DataFrame, current: pd.Series) -> Optional[Dict]:
        """Identify support/resistance bounces and store levels"""
        # Find pivot highs and lows
        highs = data['high'].values
        lows = data['low'].values
        close = current['close']
        
        # Calculate pivot points
        pivot_highs = []
        pivot_lows = []
        
        for i in range(5, len(data) - 5):
            # Pivot high: higher than 5 bars on each side
            if highs[i] == max(highs[i-5:i+6]):
                pivot_highs.append(highs[i])
            
            # Pivot low: lower than 5 bars on each side
            if lows[i] == min(lows[i-5:i+6]):
                pivot_lows.append(lows[i])
        
        if not pivot_highs or not pivot_lows:
            return None
        
        # Find nearest resistance and support
        resistances = [h for h in pivot_highs if h > close]
        supports = [l for l in pivot_lows if l < close]
        
        if not resistances or not supports:
            return None
        
        nearest_resistance = min(resistances)
        nearest_support = max(supports)
        
        # Store levels for stop/TP calculation
        self.last_analysis['support_levels'] = sorted(supports, reverse=True)[:3]  # Top 3 supports
        self.last_analysis['resistance_levels'] = sorted(resistances)[:3]  # Top 3 resistances
        
        # Check if price is near S/R level
        dist_to_resistance = (nearest_resistance - close) / close
        dist_to_support = (close - nearest_support) / close
        
        # Bouncing off support (bullish)
        if dist_to_support < self.sr_sensitivity:
            self.last_analysis['entry_reason'] = 'support_bounce'
            return {
                'signal': 'BUY',
                'reason': 'Support bounce'
            }
        
        # Rejecting at resistance (bearish)
        if dist_to_resistance < self.sr_sensitivity:
            self.last_analysis['entry_reason'] = 'resistance_rejection'
            return {
                'signal': 'SELL',
                'reason': 'Resistance rejection'
            }
        
        # Breaking resistance (bullish continuation)
        if close > nearest_resistance and dist_to_resistance < self.sr_sensitivity * 2:
            self.last_analysis['entry_reason'] = 'resistance_breakout'
            return {
                'signal': 'BUY',
                'reason': 'Resistance breakout'
            }
        
        # Breaking support (bearish continuation)
        if close < nearest_support and dist_to_support < self.sr_sensitivity * 2:
            self.last_analysis['entry_reason'] = 'support_breakdown'
            return {
                'signal': 'SELL',
                'reason': 'Support breakdown'
            }
        
        return None
    
    def _analyze_candlestick_patterns(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect candlestick patterns"""
        if len(data) < 3:
            return None
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        prev2 = data.iloc[-3]
        
        open_price = current['open']
        close = current['close']
        high = current['high']
        low = current['low']
        
        body = abs(close - open_price)
        range_total = high - low
        
        if range_total == 0:
            return None
        
        body_ratio = body / range_total
        
        # PIN BAR (long wick, small body)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        # Bullish pin bar (hammer)
        if lower_wick > body * 2 and lower_wick > upper_wick * 2 and body_ratio < 0.3:
            return {
                'signal': 'BUY',
                'reason': 'Bullish pin bar'
            }
        
        # Bearish pin bar (shooting star)
        if upper_wick > body * 2 and upper_wick > lower_wick * 2 and body_ratio < 0.3:
            return {
                'signal': 'SELL',
                'reason': 'Bearish pin bar'
            }
        
        # ENGULFING PATTERN
        prev_body = abs(prev['close'] - prev['open'])
        
        # Bullish engulfing
        if (close > open_price and  # Current is bullish
            prev['close'] < prev['open'] and  # Previous is bearish
            close > prev['open'] and  # Current close above prev open
            open_price < prev['close'] and  # Current open below prev close
            body > prev_body * 1.2):  # Current body bigger
            return {
                'signal': 'BUY',
                'reason': 'Bullish engulfing'
            }
        
        # Bearish engulfing
        if (close < open_price and  # Current is bearish
            prev['close'] > prev['open'] and  # Previous is bullish
            close < prev['open'] and  # Current close below prev open
            open_price > prev['close'] and  # Current open above prev close
            body > prev_body * 1.2):  # Current body bigger
            return {
                'signal': 'SELL',
                'reason': 'Bearish engulfing'
            }
        
        # INSIDE BAR (consolidation/breakout setup)
        if high < prev['high'] and low > prev['low']:
            # Wait for breakout
            if close > prev['high']:
                return {
                    'signal': 'BUY',
                    'reason': 'Inside bar breakout up'
                }
            elif close < prev['low']:
                return {
                    'signal': 'SELL',
                    'reason': 'Inside bar breakout down'
                }
        
        return None
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze trend and market structure, store swing points"""
        if len(data) < 20:
            return None
        
        recent = data.tail(20)
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                swing_highs.append(highs[i])
            if lows[i] == min(lows[i-2:i+3]):
                swing_lows.append(lows[i])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Store most recent swing points for stop placement
        self.last_analysis['swing_high'] = swing_highs[-1]
        self.last_analysis['swing_low'] = swing_lows[-1]
        
        # UPTREND: Higher highs and higher lows
        if (swing_highs[-1] > swing_highs[-2] and 
            swing_lows[-1] > swing_lows[-2]):
            return {
                'signal': 'BUY',
                'reason': 'Uptrend structure'
            }
        
        # DOWNTREND: Lower highs and lower lows
        if (swing_highs[-1] < swing_highs[-2] and 
            swing_lows[-1] < swing_lows[-2]):
            return {
                'signal': 'SELL',
                'reason': 'Downtrend structure'
            }
        
        # TREND BREAK
        # Bullish: Making higher high after lower low
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-2] < swing_lows[-3]:
            return {
                'signal': 'BUY',
                'reason': 'Bullish trend break'
            }
        
        # Bearish: Making lower low after higher high
        if swing_lows[-1] < swing_lows[-2] and swing_highs[-2] > swing_highs[-3]:
            return {
                'signal': 'SELL',
                'reason': 'Bearish trend break'
            }
        
        return None
    
    def _analyze_volume(self, data: pd.DataFrame) -> Optional[Dict]:
        """Volume confirmation (if available)"""
        if 'volume' not in data.columns:
            return None
        
        recent = data.tail(10)
        current = recent.iloc[-1]
        avg_volume = recent['volume'].mean()
        
        if current['volume'] < avg_volume * 0.5:
            return None  # Low volume, no conviction
        
        # High volume with price movement
        body = abs(current['close'] - current['open'])
        range_total = current['high'] - current['low']
        
        if range_total == 0:
            return None
        
        if current['volume'] > avg_volume * 1.5:
            if current['close'] > current['open']:
                return {
                    'signal': 'BUY',
                    'reason': 'High volume buying'
                }
            else:
                return {
                    'signal': 'SELL',
                    'reason': 'High volume selling'
                }
        
        return None
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Optional[Dict]:
        """Price momentum analysis"""
        if len(data) < 10:
            return None
        
        recent = data.tail(10)
        current = recent.iloc[-1]
        
        # Rate of change
        roc_5 = (current['close'] - recent.iloc[-6]['close']) / recent.iloc[-6]['close']
        roc_10 = (current['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']
        
        # Strong momentum
        if roc_5 > 0.02 and roc_10 > 0.03:  # 2% in 5 bars, 3% in 10 bars
            return {
                'signal': 'BUY',
                'reason': 'Strong bullish momentum'
            }
        
        if roc_5 < -0.02 and roc_10 < -0.03:
            return {
                'signal': 'SELL',
                'reason': 'Strong bearish momentum'
            }
        
        # Momentum reversal (divergence-like)
        if roc_5 > 0.01 and roc_10 < -0.01:  # Short term up, long term down
            return {
                'signal': 'BUY',
                'reason': 'Momentum reversal up'
            }
        
        if roc_5 < -0.01 and roc_10 > 0.01:  # Short term down, long term up
            return {
                'signal': 'SELL',
                'reason': 'Momentum reversal down'
            }
        
        return None
    
    def _calculate_structure_levels(self, data: pd.DataFrame, signal: str) -> tuple:
        """
        Calculate stop loss and take profit based on actual market structure
        Returns: (stop_loss_price, take_profit_price)
        """
        current_price = data.iloc[-1]['close']
        
        if signal == 'NEUTRAL':
            return None, None
        
        # Get stored structure levels
        supports = self.last_analysis.get('support_levels', [])
        resistances = self.last_analysis.get('resistance_levels', [])
        swing_high = self.last_analysis.get('swing_high')
        swing_low = self.last_analysis.get('swing_low')
        entry_reason = self.last_analysis.get('entry_reason')
        
        stop_loss = None
        take_profit = None
        
        if signal == 'BUY':
            # === STOP LOSS for BUY ===
            # Priority 1: Below the level we're bouncing from
            if entry_reason == 'support_bounce' and supports:
                # Stop below support with small buffer (0.2%)
                stop_loss = supports[0] * 0.998
            
            # Priority 2: Below recent swing low
            elif swing_low:
                stop_loss = swing_low * 0.998
            
            # Priority 3: Below nearest support
            elif supports:
                stop_loss = supports[0] * 0.998
            
            # Fallback: Use ATR-based (1.5% minimum)
            else:
                stop_loss = current_price * 0.985
            
            # === TAKE PROFIT for BUY ===
            # Priority 1: At next resistance level
            if resistances:
                # Target just before resistance (avoid getting rejected)
                take_profit = resistances[0] * 0.998
            
            # Priority 2: At measured move (pattern height projection)
            elif swing_low and swing_high:
                pattern_height = swing_high - swing_low
                take_profit = current_price + pattern_height
            
            # Priority 3: Use risk:reward (minimum 1.5:1)
            else:
                risk = current_price - stop_loss
                take_profit = current_price + (risk * 2.0)
            
        elif signal == 'SELL':
            # === STOP LOSS for SELL ===
            # Priority 1: Above the level we're rejecting from
            if entry_reason == 'resistance_rejection' and resistances:
                # Stop above resistance with small buffer (0.2%)
                stop_loss = resistances[0] * 1.002
            
            # Priority 2: Above recent swing high
            elif swing_high:
                stop_loss = swing_high * 1.002
            
            # Priority 3: Above nearest resistance
            elif resistances:
                stop_loss = resistances[0] * 1.002
            
            # Fallback: Use ATR-based (1.5% minimum)
            else:
                stop_loss = current_price * 1.015
            
            # === TAKE PROFIT for SELL ===
            # Priority 1: At next support level
            if supports:
                # Target just before support
                take_profit = supports[0] * 1.002
            
            # Priority 2: At measured move
            elif swing_low and swing_high:
                pattern_height = swing_high - swing_low
                take_profit = current_price - pattern_height
            
            # Priority 3: Use risk:reward (minimum 1.5:1)
            else:
                risk = stop_loss - current_price
                take_profit = current_price - (risk * 2.0)
        
        return stop_loss, take_profit
    
    def get_structure_stop_loss(self, entry_price: float, direction: str) -> Optional[float]:
        """
        Public method to get structure-based stop loss
        Called by risk manager for compatibility
        """
        if direction == 'BUY':
            swing_low = self.last_analysis.get('swing_low')
            supports = self.last_analysis.get('support_levels', [])
            
            if supports:
                return supports[0] * 0.998
            elif swing_low:
                return swing_low * 0.998
            else:
                return entry_price * 0.985  # 1.5% fallback
        
        else:  # SELL
            swing_high = self.last_analysis.get('swing_high')
            resistances = self.last_analysis.get('resistance_levels', [])
            
            if resistances:
                return resistances[0] * 1.002
            elif swing_high:
                return swing_high * 1.002
            else:
                return entry_price * 1.015  # 1.5% fallback
    
    def get_structure_take_profit(self, entry_price: float, direction: str) -> Optional[float]:
        """
        Public method to get structure-based take profit
        Called by risk manager for compatibility
        """
        if direction == 'BUY':
            resistances = self.last_analysis.get('resistance_levels', [])
            swing_high = self.last_analysis.get('swing_high')
            swing_low = self.last_analysis.get('swing_low')
            
            if resistances:
                return resistances[0] * 0.998
            elif swing_high and swing_low:
                pattern_height = swing_high - swing_low
                return entry_price + pattern_height
            else:
                return entry_price * 1.03  # 3% fallback
        
        else:  # SELL
            supports = self.last_analysis.get('support_levels', [])
            swing_high = self.last_analysis.get('swing_high')
            swing_low = self.last_analysis.get('swing_low')
            
            if supports:
                return supports[0] * 1.002
            elif swing_high and swing_low:
                pattern_height = swing_high - swing_low
                return entry_price - pattern_height
            else:
                return entry_price * 0.97  # 3% fallback
