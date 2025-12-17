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
        
        # Symbol-specific stop loss multipliers (ATR multiplier)
        # PHILOSOPHY: Wide stops, big targets = high R:R
        self.stop_multipliers = {
            'R_100': 2.0,  # Wide stops for clean breakouts
            '1HZ10V': 1.0,  # Give trades room to breathe
            '1HZ50V': 1.5,  # Not used anymore but keeping for reference
        }
        
        # Multi-timeframe settings
        self.timeframes = {
            'short': 50,    # ~50 ticks (micro structure)
            'medium': 100,  # ~100 ticks (entry timeframe)
            'long': 200     # ~200 ticks (trend context)
        }
        
        # Store structure levels per symbol for stop/TP placement
        self.symbol_analysis = {}  # Dict of {symbol: analysis_data}
        
    def _get_stop_multiplier(self, symbol: str) -> float:
        """Get the appropriate stop loss multiplier for a symbol"""
        return self.stop_multipliers.get(symbol, 0.5)  # Default to 0.5 ATR
        
    def analyze(self, data: pd.DataFrame, symbol: str = 'UNKNOWN') -> Dict:
        """Analyze price action and generate signals with multi-timeframe confirmation"""
        if len(data) < self.timeframes['long']:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        # Initialize analysis storage for this symbol
        if symbol not in self.symbol_analysis:
            self.symbol_analysis[symbol] = {
                'support_levels': [],
                'resistance_levels': [],
                'swing_high': None,
                'swing_low': None,
                'entry_reason': None
            }
        
        # Store current symbol for structure level methods
        self.current_symbol = symbol
        
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
        self.symbol_analysis[self.current_symbol]['support_levels'] = sorted(supports, reverse=True)[:3]  # Top 3 supports
        self.symbol_analysis[self.current_symbol]['resistance_levels'] = sorted(resistances)[:3]  # Top 3 resistances
        
        # Check if price is near S/R level
        dist_to_resistance = (nearest_resistance - close) / close
        dist_to_support = (close - nearest_support) / close
        
        # Bouncing off support (bullish)
        if dist_to_support < self.sr_sensitivity:
            self.symbol_analysis[self.current_symbol]['entry_reason'] = 'support_bounce'
            return {
                'signal': 'BUY',
                'reason': 'Support bounce'
            }
        
        # Rejecting at resistance (bearish)
        if dist_to_resistance < self.sr_sensitivity:
            self.symbol_analysis[self.current_symbol]['entry_reason'] = 'resistance_rejection'
            return {
                'signal': 'SELL',
                'reason': 'Resistance rejection'
            }
        
        # Breaking resistance (bullish continuation)
        if close > nearest_resistance and dist_to_resistance < self.sr_sensitivity * 2:
            self.symbol_analysis[self.current_symbol]['entry_reason'] = 'resistance_breakout'
            return {
                'signal': 'BUY',
                'reason': 'Resistance breakout'
            }
        
        # Breaking support (bearish continuation)
        if close < nearest_support and dist_to_support < self.sr_sensitivity * 2:
            self.symbol_analysis[self.current_symbol]['entry_reason'] = 'support_breakdown'
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
        self.symbol_analysis[self.current_symbol]['swing_high'] = swing_highs[-1]
        self.symbol_analysis[self.current_symbol]['swing_low'] = swing_lows[-1]
        
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
        
        # TREND BREAK (need at least 3-4 swing points)
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
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
        
        # Get stored structure levels for current symbol
        analysis = self.symbol_analysis.get(self.current_symbol, {})
        supports = analysis.get('support_levels', [])
        resistances = analysis.get('resistance_levels', [])
        swing_high = analysis.get('swing_high')
        swing_low = analysis.get('swing_low')
        entry_reason = analysis.get('entry_reason')
        
        # Get ATR for dynamic stop sizing
        atr = data.iloc[-1].get('atr', current_price * 0.02)  # Fallback 2%
        
        # Get symbol-specific stop multiplier
        stop_multiplier = self._get_stop_multiplier(self.current_symbol)
        
        stop_loss = None
        take_profit = None
        
        if signal == 'BUY':
            # === STOP LOSS for BUY ===
            # Strategy: Place stop below structure with ATR buffer
            
            # Find nearest support below current price
            nearest_support = None
            for s in reversed(supports):
                if s < current_price:
                    nearest_support = s
                    break
            
            # Calculate stop options
            structure_stop = None
            if entry_reason == 'support_bounce' and nearest_support:
                # Stop below bounce level with ATR buffer
                structure_stop = nearest_support - (atr * stop_multiplier)
            elif nearest_support:
                # Stop below nearest support with ATR buffer  
                structure_stop = nearest_support - (atr * stop_multiplier)
            elif swing_low and swing_low < current_price:
                # Stop below swing low with ATR buffer
                structure_stop = swing_low - (atr * stop_multiplier)
            
            # Use ATR-based stop if no structure or structure too far
            atr_stop = current_price - (atr * stop_multiplier)  # Symbol-specific ATR stop
            
            if structure_stop:
                # Use structure stop if it's tighter than 1.0 ATR (sniper precision)
                if (current_price - structure_stop) <= (atr * 1.0):
                    stop_loss = structure_stop
                else:
                    # Structure too far, use ATR stop
                    stop_loss = atr_stop
            else:
                stop_loss = atr_stop
            
            # Enforce broker's minimum stop distance
            symbol_info = self.api.get_symbol_info(self.current_symbol)
            if symbol_info and 'min_stop_distance' in symbol_info:
                min_distance = symbol_info['min_stop_distance']
                min_stop = current_price - (min_distance * 1.1)  # 10% buffer
                if stop_loss < min_stop:
                    logger.warning(f"⚠️ Adjusting stop from {stop_loss:.5f} to {min_stop:.5f} (broker minimum: {min_distance:.5f})")
                    stop_loss = min_stop
            
            # === TAKE PROFIT for BUY ===
            # Strategy: ONLY use key resistance levels, no arbitrary R:R
            
            risk = current_price - stop_loss
            
            # Find nearest resistance above current price
            nearest_resistance = None
            for r in resistances:
                if r > current_price:
                    nearest_resistance = r
                    break
            
            # STRICT: Only take trade if there's a clear resistance target
            if nearest_resistance:
                take_profit = nearest_resistance * 0.998  # Just before resistance
                
                # Validate it gives at least 3:1 R:R (quality over quantity)
                reward = take_profit - current_price
                if reward / risk >= 3.0:
                    # Excellent trade: clear target with high R:R
                    pass
                else:
                    # Target too close for high R:R, skip trade
                    take_profit = None
            
            # Try measured move only if strong pattern
            elif swing_high and swing_low:
                pattern_height = swing_high - swing_low
                # Only if pattern is significant (> 2x risk)
                if pattern_height > (risk * 2.0):
                    take_profit = current_price + pattern_height
                else:
                    take_profit = None  # Pattern too small
            else:
                # No clear target = no trade
                take_profit = None
            
        elif signal == 'SELL':
            # === STOP LOSS for SELL ===
            # Strategy: Place stop above structure with ATR buffer
            
            # Find nearest resistance above current price
            nearest_resistance = None
            for r in resistances:
                if r > current_price:
                    nearest_resistance = r
                    break
            
            # Calculate stop options
            structure_stop = None
            if entry_reason == 'resistance_rejection' and nearest_resistance:
                # Stop above rejection level with ATR buffer
                structure_stop = nearest_resistance + (atr * stop_multiplier)
            elif nearest_resistance:
                # Stop above nearest resistance with ATR buffer
                structure_stop = nearest_resistance + (atr * stop_multiplier)
            elif swing_high and swing_high > current_price:
                # Stop above swing high with ATR buffer
                structure_stop = swing_high + (atr * stop_multiplier)
            
            # Use ATR-based stop if no structure or structure too far
            atr_stop = current_price + (atr * stop_multiplier)  # Symbol-specific ATR stop
            
            if structure_stop:
                # Use structure stop if it's tighter than 1.0 ATR (sniper precision)
                if (structure_stop - current_price) <= (atr * 1.0):
                    stop_loss = structure_stop
                else:
                    # Structure too far, use ATR stop
                    stop_loss = atr_stop
            else:
                stop_loss = atr_stop
            
            # Enforce broker's minimum stop distance
            symbol_info = self.api.get_symbol_info(self.current_symbol)
            if symbol_info and 'min_stop_distance' in symbol_info:
                min_distance = symbol_info['min_stop_distance']
                max_stop = current_price + (min_distance * 1.1)  # 10% buffer
                if stop_loss > max_stop:
                    logger.warning(f"⚠️ Adjusting stop from {stop_loss:.5f} to {max_stop:.5f} (broker minimum: {min_distance:.5f})")
                    stop_loss = max_stop
            
            # === TAKE PROFIT for SELL ===
            # Strategy: ONLY use key support levels, no arbitrary R:R
            
            risk = stop_loss - current_price
            
            # Find nearest support below current price
            nearest_support = None
            for s in reversed(supports):
                if s < current_price:
                    nearest_support = s
                    break
            
            # STRICT: Only take trade if there's a clear support target
            if nearest_support:
                take_profit = nearest_support * 1.002  # Just before support
                
                # Validate it gives at least 3:1 R:R (quality over quantity)
                reward = current_price - take_profit
                if reward / risk >= 3.0:
                    # Excellent trade: clear target with high R:R
                    pass
                else:
                    # Target too close for high R:R, skip trade
                    take_profit = None
            
            # Try measured move only if strong pattern
            elif swing_high and swing_low:
                pattern_height = swing_high - swing_low
                # Only if pattern is significant (> 2x risk)
                if pattern_height > (risk * 2.0):
                    take_profit = current_price - pattern_height
                else:
                    take_profit = None  # Pattern too small
            else:
                # No clear target = no trade
                take_profit = None
        
        return stop_loss, take_profit
        
        stop_loss = None
        take_profit = None
        
        if signal == 'BUY':
            # === STOP LOSS for BUY ===
            # Priority 1: Below the level we're bouncing from
            if entry_reason == 'support_bounce' and supports:
                # Find the support we're bouncing from (highest support below current price)
                bounce_support = None
                for s in reversed(supports):
                    if s < current_price:
                        bounce_support = s
                        break
                if bounce_support:
                    # Stop below support with small buffer (0.2%)
                    stop_loss = bounce_support * 0.998
            
            # Priority 2: Below recent swing low
            if not stop_loss and swing_low and swing_low < current_price:
                stop_loss = swing_low * 0.998
            
            # Priority 3: Below nearest support below current price
            if not stop_loss and supports:
                nearest_support = None
                for s in reversed(supports):
                    if s < current_price:
                        nearest_support = s
                        break
                if nearest_support:
                    stop_loss = nearest_support * 0.998
            
            # Fallback: Use ATR-based (1.5% minimum)
            if not stop_loss:
                stop_loss = current_price * 0.985
            
            # === TAKE PROFIT for BUY ===
            # Priority 1: At next resistance level ABOVE current price
            if resistances:
                # Find first resistance above current price
                next_resistance = None
                for r in resistances:
                    if r > current_price:
                        next_resistance = r
                        break
                
                if next_resistance:
                    # Target just before resistance (avoid getting rejected)
                    take_profit = next_resistance * 0.998
                else:
                    # No resistance above, use measured move
                    if swing_low and swing_high:
                        pattern_height = swing_high - swing_low
                        take_profit = current_price + pattern_height
                    else:
                        risk = current_price - stop_loss
                        take_profit = current_price + (risk * 2.0)
            
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
                # Find the resistance we're rejecting from (lowest resistance above current price)
                rejection_resistance = None
                for r in resistances:
                    if r > current_price:
                        rejection_resistance = r
                        break
                if rejection_resistance:
                    # Stop above resistance with small buffer (0.2%)
                    stop_loss = rejection_resistance * 1.002
            
            # Priority 2: Above recent swing high
            if not stop_loss and swing_high and swing_high > current_price:
                stop_loss = swing_high * 1.002
            
            # Priority 3: Above nearest resistance above current price
            if not stop_loss and resistances:
                nearest_resistance = None
                for r in resistances:
                    if r > current_price:
                        nearest_resistance = r
                        break
                if nearest_resistance:
                    stop_loss = nearest_resistance * 1.002
            
            # Fallback: Use ATR-based (1.5% minimum)
            if not stop_loss:
                stop_loss = current_price * 1.015
            
            # === TAKE PROFIT for SELL ===
            # Priority 1: At next support level BELOW current price
            if supports:
                # Find first support below current price (iterate backwards since supports are sorted low to high)
                next_support = None
                for s in reversed(supports):
                    if s < current_price:
                        next_support = s
                        break
                
                if next_support:
                    # Target just before support
                    take_profit = next_support * 1.002
                else:
                    # No support below, use measured move
                    if swing_low and swing_high:
                        pattern_height = swing_high - swing_low
                        take_profit = current_price - pattern_height
                    else:
                        risk = stop_loss - current_price
                        take_profit = current_price - (risk * 2.0)
            
            # Priority 2: At measured move
            elif swing_low and swing_high:
                pattern_height = swing_high - swing_low
                take_profit = current_price - pattern_height
            
            # Priority 3: Use risk:reward (minimum 1.5:1)
            else:
                risk = stop_loss - current_price
                take_profit = current_price - (risk * 2.0)
        
        return stop_loss, take_profit
    
    def get_structure_stop_loss(self, entry_price: float, direction: str, symbol: str = None) -> Optional[float]:
        """
        Public method to get structure-based stop loss with ATR consideration
        Uses 2 ATR or structure (whichever is tighter, max 3 ATR)
        """
        # Use current_symbol if symbol not provided
        if symbol is None:
            symbol = getattr(self, 'current_symbol', 'UNKNOWN')
        
        analysis = self.symbol_analysis.get(symbol, {})
        
        # Get ATR (estimate if not available)
        atr = entry_price * 0.02  # 2% fallback
        
        if direction == 'BUY':
            supports = analysis.get('support_levels', [])
            swing_low = analysis.get('swing_low')
            
            # Find nearest support below entry
            nearest_support = None
            for s in reversed(supports):
                if s < entry_price:
                    nearest_support = s
                    break
            
            # ATR-based stop (0.5 ATR for sniper entries)
            atr_stop = entry_price - (atr * 0.5)
            
            # Structure stop with ATR buffer
            if nearest_support:
                structure_stop = nearest_support - (atr * 0.5)
                # Use structure if within 1.0 ATR
                if (entry_price - structure_stop) <= (atr * 1.0):
                    return max(structure_stop, entry_price * 0.999)  # Min 0.1%
            elif swing_low and swing_low < entry_price:
                structure_stop = swing_low - (atr * 0.5)
                if (entry_price - structure_stop) <= (atr * 1.0):
                    return max(structure_stop, entry_price * 0.999)
            
            # Default to 0.5 ATR
            return max(atr_stop, entry_price * 0.999)
        
        else:  # SELL
            resistances = analysis.get('resistance_levels', [])
            swing_high = analysis.get('swing_high')
            
            # Find nearest resistance above entry
            nearest_resistance = None
            for r in resistances:
                if r > entry_price:
                    nearest_resistance = r
                    break
            
            # ATR-based stop (0.5 ATR for sniper entries)
            atr_stop = entry_price + (atr * 0.5)
            
            # Structure stop with ATR buffer
            if nearest_resistance:
                structure_stop = nearest_resistance + (atr * 0.5)
                # Use structure if within 1.0 ATR
                if (structure_stop - entry_price) <= (atr * 1.0):
                    return min(structure_stop, entry_price * 1.001)  # Max 0.1%
            elif swing_high and swing_high > entry_price:
                structure_stop = swing_high + (atr * 0.5)
                if (structure_stop - entry_price) <= (atr * 1.0):
                    return min(structure_stop, entry_price * 1.001)
            
            # Default to 0.5 ATR
            return min(atr_stop, entry_price * 1.001)
    
    def get_structure_take_profit(self, entry_price: float, direction: str, symbol: str = None) -> Optional[float]:
        """
        Public method to get structure-based take profit
        Ensures minimum 2:1 R:R ratio
        """
        # Use current_symbol if symbol not provided
        if symbol is None:
            symbol = getattr(self, 'current_symbol', 'UNKNOWN')
        
        analysis = self.symbol_analysis.get(symbol, {})
        
        # Calculate stop loss to determine risk
        stop_loss = self.get_structure_stop_loss(entry_price, direction, symbol)
        risk = abs(entry_price - stop_loss)
        min_reward = risk * 1.5  # Minimum 1.5:1 R:R
        
        if direction == 'BUY':
            resistances = analysis.get('resistance_levels', [])
            swing_high = analysis.get('swing_high')
            swing_low = analysis.get('swing_low')
            
            min_target = entry_price + min_reward
            
            # Find nearest resistance above entry
            nearest_resistance = None
            for r in resistances:
                if r > entry_price:
                    nearest_resistance = r
                    break
            
            # Use resistance if it provides 2:1+ R:R
            if nearest_resistance and nearest_resistance >= min_target:
                return nearest_resistance * 0.998
            
            # Try measured move
            if swing_high and swing_low:
                pattern_height = swing_high - swing_low
                measured_target = entry_price + pattern_height
                if measured_target >= min_target:
                    return measured_target
            
            # Default to 2:1 R:R
            return min_target
        
        else:  # SELL
            supports = analysis.get('support_levels', [])
            swing_high = analysis.get('swing_high')
            swing_low = analysis.get('swing_low')
            
            min_target = entry_price - min_reward
            
            # Find nearest support below entry
            nearest_support = None
            for s in reversed(supports):
                if s < entry_price:
                    nearest_support = s
                    break
            
            # Use support if it provides 2:1+ R:R
            if nearest_support and nearest_support <= min_target:
                return nearest_support * 1.002
            
            # Try measured move
            if swing_high and swing_low:
                pattern_height = swing_high - swing_low
                measured_target = entry_price - pattern_height
                if measured_target <= min_target:
                    return measured_target
            
            # Default to 2:1 R:R
            return min_target

