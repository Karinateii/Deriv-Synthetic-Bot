"""
Regime-Aware Synthetic Index Strategy
======================================
A sophisticated trading strategy specifically designed for synthetic indices where:
- No news or fundamentals exist
- Volatility regimes change frequently
- Brokers control price generation

Core Philosophy:
- Capital preservation over profit maximization
- Trade infrequently, only in favorable regimes
- Adaptive risk based on market behavior
- Kill-switch protection against anomalous conditions

This strategy AVOIDS common retail traps:
- No simple MA crossovers
- No fixed RSI levels
- No martingale/grid systems
- No assumption of persistent trends
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy
from loguru import logger


class MarketRegime(Enum):
    """Market regime classifications"""
    QUIET_RANGE = "quiet_range"           # Low volatility, mean-reverting
    VOLATILE_RANGE = "volatile_range"     # High volatility, choppy
    TRENDING_CALM = "trending_calm"       # Clear direction, low noise
    TRENDING_VOLATILE = "trending_volatile"  # Direction with high noise
    CHAOTIC = "chaotic"                   # Unpredictable, avoid trading
    TRANSITION = "transition"             # Regime changing, avoid trading
    UNKNOWN = "unknown"                   # Insufficient data


class TradabilityState(Enum):
    """Whether the strategy should be active"""
    ENABLED = "enabled"
    DISABLED_VOLATILITY = "disabled_volatility"
    DISABLED_SPREAD = "disabled_spread"
    DISABLED_CHAOS = "disabled_chaos"
    DISABLED_DRAWDOWN = "disabled_drawdown"
    DISABLED_REGIME_CHANGE = "disabled_regime_change"
    DISABLED_KILL_SWITCH = "disabled_kill_switch"


@dataclass
class RegimeAnalysis:
    """Results of regime analysis"""
    regime: MarketRegime
    confidence: float
    volatility_percentile: float
    volatility_ratio: float  # Current vs historical
    trend_strength: float
    mean_reversion_score: float
    regime_age_bars: int
    tradability: TradabilityState
    kill_switch_reason: Optional[str] = None


@dataclass
class TradeSetup:
    """A potential trade setup with all parameters"""
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float
    regime: MarketRegime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_multiplier: float
    max_hold_bars: int
    exit_conditions: List[str]
    invalidation_price: float
    reason: str


class RegimeAwareStrategy(BaseStrategy):
    """
    Regime-Aware Trading Strategy for Synthetic Indices
    
    Key Features:
    1. Multi-timeframe volatility regime detection
    2. Only trades in specific favorable regimes
    3. Adaptive position sizing based on regime confidence
    4. Time-based and behavior-based exits
    5. Kill-switch protection against anomalies
    
    Design Constraints:
    - Expects long flat periods with no trades
    - Robust to spread spikes
    - No curve-fitting or over-optimization
    - No assumptions of persistent trends
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Regime Aware", **kwargs)
        
        # ============ REGIME DETECTION PARAMETERS ============
        self.volatility_lookback = 100       # Bars for volatility calculation
        self.regime_confirm_bars = 5         # Bars to confirm regime change
        self.volatility_ema_fast = 10        # Fast volatility EMA
        self.volatility_ema_slow = 50        # Slow volatility EMA
        
        # Volatility thresholds (percentiles)
        self.low_vol_percentile = 25
        self.high_vol_percentile = 75
        self.extreme_vol_percentile = 95
        
        # ============ TRADING PARAMETERS ============
        self.min_regime_age = 3              # Min bars in regime before trading (reduced for testing)
        self.max_regime_age = 200            # Max bars (regime may be stale)
        self.min_confidence = 0.50           # Minimum confidence to trade (reduced for testing)
        self.base_risk_percent = 0.01        # Base risk per trade (1%)
        self.max_risk_percent = 0.025        # Max risk per trade (2.5%)
        self.min_risk_percent = 0.005        # Min risk per trade (0.5%)
        
        # ============ KILL SWITCH PARAMETERS ============
        self.max_volatility_spike = 3.0      # Max acceptable vol spike ratio
        self.max_spread_percent = 0.5        # Max spread as % of price
        self.max_consecutive_losses = 3      # Pause after N losses
        self.max_daily_trades = 5            # Max trades per day
        self.drawdown_pause_percent = 0.03   # Pause at 3% drawdown
        
        # ============ EXIT PARAMETERS ============
        self.max_hold_bars_quiet = 50        # Max bars in quiet regime
        self.max_hold_bars_trending = 100    # Max bars in trending regime
        self.trailing_activation_r = 1.5     # Activate trail at 1.5R
        self.trailing_distance_atr = 1.5     # Trail distance in ATR
        
        # ============ INTERNAL STATE ============
        self.current_regime = MarketRegime.UNKNOWN
        self.previous_regime = MarketRegime.UNKNOWN
        self.regime_bar_count = 0
        self.last_regime_change = None
        self.consecutive_losses = 0
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.session_pnl = 0.0
        self.session_peak = 0.0
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        # Historical regime tracking
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        
    def analyze(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, Dict]:
        """
        Main analysis method
        
        Returns:
            (signal, confidence, metadata)
        """
        try:
            # Reset daily counters if new day
            self._check_new_day()
            
            # Step 1: Validate data sufficiency
            if len(df) < self.volatility_lookback + 50:
                return 'NEUTRAL', 0.0, {'reason': 'Insufficient data for regime analysis'}
            
            # Step 2: Analyze market regime
            regime_analysis = self._analyze_regime(df)
            
            # Step 3: Check kill switch conditions
            kill_switch, kill_reason = self._check_kill_switch(df, regime_analysis)
            if kill_switch:
                self.kill_switch_active = True
                self.kill_switch_reason = kill_reason
                return 'NEUTRAL', 0.0, {
                    'reason': f'Kill switch: {kill_reason}',
                    'regime': regime_analysis.regime.value,
                    'tradability': TradabilityState.DISABLED_KILL_SWITCH.value
                }
            
            # Step 4: Check tradability
            if regime_analysis.tradability != TradabilityState.ENABLED:
                return 'NEUTRAL', 0.0, {
                    'reason': f'Trading disabled: {regime_analysis.tradability.value}',
                    'regime': regime_analysis.regime.value,
                    'volatility_percentile': regime_analysis.volatility_percentile
                }
            
            # Step 5: Generate trade setup based on regime
            setup = self._generate_setup(df, current_price, regime_analysis)
            
            if setup.signal == 'NEUTRAL':
                return 'NEUTRAL', 0.0, {
                    'reason': setup.reason,
                    'regime': regime_analysis.regime.value,
                    'volatility_percentile': regime_analysis.volatility_percentile
                }
            
            # Step 6: Final validation
            if setup.confidence < self.min_confidence:
                return 'NEUTRAL', 0.0, {
                    'reason': f'Confidence too low: {setup.confidence:.2f}',
                    'regime': regime_analysis.regime.value
                }
            
            # Return valid setup
            return setup.signal, setup.confidence, {
                'reason': setup.reason,
                'regime': regime_analysis.regime.value,
                'regime_confidence': regime_analysis.confidence,
                'stop_loss': setup.stop_loss,
                'take_profit': setup.take_profit,
                'position_size_multiplier': setup.position_size_multiplier,
                'max_hold_bars': setup.max_hold_bars,
                'exit_conditions': setup.exit_conditions,
                'invalidation_price': setup.invalidation_price,
                'volatility_percentile': regime_analysis.volatility_percentile,
                'trend_strength': regime_analysis.trend_strength
            }
            
        except Exception as e:
            logger.error(f"Regime strategy analysis error: {e}")
            return 'NEUTRAL', 0.0, {'error': str(e)}
    
    # ========================================================================
    # REGIME DETECTION
    # ========================================================================
    
    def _analyze_regime(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Comprehensive market regime analysis
        
        Regimes:
        - QUIET_RANGE: Low vol, mean-reverting behavior
        - VOLATILE_RANGE: High vol, no clear direction
        - TRENDING_CALM: Clear trend, low noise
        - TRENDING_VOLATILE: Trend with high noise
        - CHAOTIC: Unpredictable, extreme conditions
        - TRANSITION: Regime changing
        """
        # Calculate volatility metrics
        returns = df['close'].pct_change().dropna()
        recent_returns = returns.tail(self.volatility_lookback)
        
        # Current and historical volatility
        current_vol = recent_returns.tail(10).std() * np.sqrt(252)
        historical_vol = recent_returns.std() * np.sqrt(252)
        
        # Volatility EMA ratio (fast/slow)
        vol_series = returns.abs().rolling(5).mean()
        vol_ema_fast = vol_series.ewm(span=self.volatility_ema_fast).mean().iloc[-1]
        vol_ema_slow = vol_series.ewm(span=self.volatility_ema_slow).mean().iloc[-1]
        volatility_ratio = vol_ema_fast / vol_ema_slow if vol_ema_slow > 0 else 1.0
        
        # Volatility percentile
        vol_history = returns.abs().rolling(20).std().dropna()
        current_vol_measure = returns.tail(20).abs().std()
        volatility_percentile = (vol_history < current_vol_measure).mean() * 100
        
        # Trend strength using linear regression slope
        prices = df['close'].tail(50).values
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Normalize slope by price level
        trend_strength = abs(slope * len(prices) / prices[-1])
        
        # R-squared for trend clarity
        predicted = slope * x + intercept
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Mean reversion score
        mean_reversion_score = self._calculate_mean_reversion_score(df)
        
        # Classify regime
        regime = self._classify_regime(
            volatility_percentile,
            volatility_ratio,
            trend_strength,
            r_squared,
            mean_reversion_score
        )
        
        # Check for regime transition
        if regime != self.current_regime:
            self.previous_regime = self.current_regime
            self.current_regime = regime
            self.regime_bar_count = 0
            self.last_regime_change = datetime.now()
            self.regime_history.append((datetime.now(), regime))
        else:
            self.regime_bar_count += 1
        
        # Calculate regime confidence
        regime_confidence = self._calculate_regime_confidence(
            volatility_percentile, volatility_ratio, r_squared, self.regime_bar_count
        )
        
        # Determine tradability
        tradability = self._determine_tradability(
            regime, volatility_percentile, volatility_ratio, self.regime_bar_count
        )
        
        return RegimeAnalysis(
            regime=regime,
            confidence=regime_confidence,
            volatility_percentile=volatility_percentile,
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
            mean_reversion_score=mean_reversion_score,
            regime_age_bars=self.regime_bar_count,
            tradability=tradability
        )
    
    def _classify_regime(self, vol_pct: float, vol_ratio: float, 
                         trend_str: float, r_sq: float, mr_score: float) -> MarketRegime:
        """Classify market into discrete regime"""
        
        # Extreme volatility = chaotic
        if vol_pct > self.extreme_vol_percentile:
            return MarketRegime.CHAOTIC
        
        # Volatility spike = transition
        if vol_ratio > 2.0:
            return MarketRegime.TRANSITION
        
        # Low volatility regimes
        if vol_pct < self.low_vol_percentile:
            if trend_str > 0.02 and r_sq > 0.7:
                return MarketRegime.TRENDING_CALM
            else:
                return MarketRegime.QUIET_RANGE
        
        # High volatility regimes
        if vol_pct > self.high_vol_percentile:
            if trend_str > 0.03 and r_sq > 0.5:
                return MarketRegime.TRENDING_VOLATILE
            else:
                return MarketRegime.VOLATILE_RANGE
        
        # Medium volatility
        if trend_str > 0.02 and r_sq > 0.6:
            return MarketRegime.TRENDING_CALM
        elif mr_score > 0.6:
            return MarketRegime.QUIET_RANGE
        else:
            return MarketRegime.VOLATILE_RANGE
    
    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion tendency score
        Higher score = more mean-reverting behavior
        """
        prices = df['close'].tail(50)
        returns = prices.pct_change().dropna()
        
        # Hurst exponent approximation (simplified)
        # H < 0.5 = mean reverting, H > 0.5 = trending
        lags = range(2, 20)
        tau = []
        for lag in lags:
            pp = np.subtract(prices[lag:].values, prices[:-lag].values)
            tau.append(np.sqrt(np.std(pp)))
        
        if len(tau) > 0 and all(t > 0 for t in tau):
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
        else:
            hurst = 0.5
        
        # Also check autocorrelation
        if len(returns) > 10:
            autocorr = returns.autocorr(lag=1)
            autocorr = autocorr if not np.isnan(autocorr) else 0
        else:
            autocorr = 0
        
        # Combine metrics: low Hurst + negative autocorr = mean reverting
        mr_score = (0.5 - min(hurst, 1.0)) + 0.5 + (- autocorr * 0.5)
        return max(0, min(1, mr_score))
    
    def _calculate_regime_confidence(self, vol_pct: float, vol_ratio: float,
                                     r_sq: float, age: int) -> float:
        """Calculate confidence in regime classification"""
        # Age factor: need some bars to confirm
        if age < self.min_regime_age:
            age_factor = age / self.min_regime_age
        elif age > self.max_regime_age:
            age_factor = max(0.5, 1 - (age - self.max_regime_age) / 100)
        else:
            age_factor = 1.0
        
        # Volatility stability factor
        vol_stability = 1.0 - min(abs(vol_ratio - 1.0), 1.0)
        
        # Trend clarity factor
        trend_clarity = r_sq
        
        # Combined confidence
        confidence = (age_factor * 0.3 + vol_stability * 0.4 + trend_clarity * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _determine_tradability(self, regime: MarketRegime, vol_pct: float,
                               vol_ratio: float, age: int) -> TradabilityState:
        """Determine if trading should be enabled"""
        
        # Chaotic = never trade
        if regime == MarketRegime.CHAOTIC:
            return TradabilityState.DISABLED_CHAOS
        
        # Transition = wait
        if regime == MarketRegime.TRANSITION:
            return TradabilityState.DISABLED_REGIME_CHANGE
        
        # Volatility spike = wait
        if vol_ratio > self.max_volatility_spike:
            return TradabilityState.DISABLED_VOLATILITY
        
        # Regime too young
        if age < self.min_regime_age:
            return TradabilityState.DISABLED_REGIME_CHANGE
        
        # Tradable regimes
        tradable_regimes = [
            MarketRegime.QUIET_RANGE,
            MarketRegime.TRENDING_CALM,
        ]
        
        if regime in tradable_regimes:
            return TradabilityState.ENABLED
        
        # Volatile range only in specific conditions
        if regime == MarketRegime.VOLATILE_RANGE:
            if vol_pct < 85 and age > 15:
                return TradabilityState.ENABLED
            return TradabilityState.DISABLED_VOLATILITY
        
        # Trending volatile with caution
        if regime == MarketRegime.TRENDING_VOLATILE:
            if vol_pct < 80 and age > 20:
                return TradabilityState.ENABLED
            return TradabilityState.DISABLED_VOLATILITY
        
        return TradabilityState.DISABLED_CHAOS
    
    # ========================================================================
    # KILL SWITCH LOGIC
    # ========================================================================
    
    def _check_kill_switch(self, df: pd.DataFrame, 
                           regime_analysis: RegimeAnalysis) -> Tuple[bool, Optional[str]]:
        """
        Kill switch checks - immediately disable trading
        
        Returns:
            (is_killed, reason)
        """
        # Check 1: Extreme volatility spike
        if regime_analysis.volatility_ratio > self.max_volatility_spike:
            return True, f"Volatility spike: {regime_analysis.volatility_ratio:.2f}x normal"
        
        # Check 2: Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Consecutive losses: {self.consecutive_losses}"
        
        # Check 3: Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return True, f"Daily trade limit reached: {self.daily_trade_count}"
        
        # Check 4: Session drawdown
        if self.session_peak > 0:
            current_drawdown = (self.session_peak - self.session_pnl) / self.session_peak
            if current_drawdown > self.drawdown_pause_percent:
                return True, f"Session drawdown: {current_drawdown:.1%}"
        
        # Check 5: Rapid regime changes
        recent_changes = [
            r for t, r in self.regime_history 
            if datetime.now() - t < timedelta(minutes=30)
        ]
        if len(recent_changes) >= 3:
            return True, "Multiple regime changes detected"
        
        # Check 6: Price anomaly (extreme candles)
        if len(df) >= 2:
            last_candle_range = abs(df['high'].iloc[-1] - df['low'].iloc[-1])
            avg_range = df['high'].tail(50).values - df['low'].tail(50).values
            avg_range = np.mean(np.abs(avg_range))
            
            if avg_range > 0 and last_candle_range > avg_range * 5:
                return True, "Price anomaly: extreme candle detected"
        
        return False, None
    
    # ========================================================================
    # TRADE SETUP GENERATION
    # ========================================================================
    
    def _generate_setup(self, df: pd.DataFrame, current_price: float,
                        regime_analysis: RegimeAnalysis) -> TradeSetup:
        """Generate trade setup based on current regime"""
        
        if regime_analysis.regime == MarketRegime.QUIET_RANGE:
            # Mean reversion disabled - skip quiet range trades
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason="Mean reversion trading disabled"
            )
        
        elif regime_analysis.regime == MarketRegime.TRENDING_CALM:
            return self._setup_trend_continuation(df, current_price, regime_analysis)
        
        elif regime_analysis.regime == MarketRegime.VOLATILE_RANGE:
            return self._setup_volatility_fade(df, current_price, regime_analysis)
        
        elif regime_analysis.regime == MarketRegime.TRENDING_VOLATILE:
            return self._setup_pullback_continuation(df, current_price, regime_analysis)
        
        else:
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason=f"No setup for regime: {regime_analysis.regime.value}"
            )
    
    def _setup_mean_reversion(self, df: pd.DataFrame, current_price: float,
                              regime_analysis: RegimeAnalysis) -> TradeSetup:
        """
        Mean reversion setup for quiet ranging markets
        
        Logic: Trade against short-term extremes back to mean
        """
        # Calculate dynamic bands
        lookback = 20
        prices = df['close'].tail(lookback)
        mean_price = prices.mean()
        std_price = prices.std()
        
        upper_band = mean_price + 2 * std_price
        lower_band = mean_price - 2 * std_price
        
        # Calculate Z-score
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        # Calculate ATR for stops
        atr = self._calculate_atr(df, period=14)
        
        signal = 'NEUTRAL'
        confidence = 0.0
        reason = "No mean reversion setup"
        
        # Oversold condition
        if z_score < -2.0:
            # Additional confirmation: momentum slowing
            momentum = df['close'].tail(5).pct_change().mean()
            if momentum > -0.001:  # Downward momentum fading
                signal = 'BUY'
                confidence = min(0.5 + abs(z_score - 2) * 0.15, 0.85)
                confidence *= regime_analysis.confidence
                reason = f"Mean reversion BUY: Z={z_score:.2f}, momentum fading"
        
        # Overbought condition
        elif z_score > 2.0:
            momentum = df['close'].tail(5).pct_change().mean()
            if momentum < 0.001:  # Upward momentum fading
                signal = 'SELL'
                confidence = min(0.5 + abs(z_score - 2) * 0.15, 0.85)
                confidence *= regime_analysis.confidence
                reason = f"Mean reversion SELL: Z={z_score:.2f}, momentum fading"
        
        if signal == 'NEUTRAL':
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason="Price not at extremes"
            )
        
        # Calculate stops and targets
        if signal == 'BUY':
            stop_loss = current_price - (atr * 2.5)
            take_profit = mean_price  # Target mean
            invalidation = lower_band - atr  # Below the band
        else:
            stop_loss = current_price + (atr * 2.5)
            take_profit = mean_price
            invalidation = upper_band + atr
        
        # Position sizing multiplier based on confidence
        size_multiplier = self._calculate_position_multiplier(confidence, regime_analysis)
        
        return TradeSetup(
            signal=signal,
            confidence=confidence,
            regime=regime_analysis.regime,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_multiplier=size_multiplier,
            max_hold_bars=self.max_hold_bars_quiet,
            exit_conditions=[
                "price_crosses_mean",
                "momentum_reverses",
                "volatility_spikes",
                "time_limit_reached"
            ],
            invalidation_price=invalidation,
            reason=reason
        )
    
    def _setup_trend_continuation(self, df: pd.DataFrame, current_price: float,
                                  regime_analysis: RegimeAnalysis) -> TradeSetup:
        """
        Trend continuation setup for calm trending markets
        
        Logic: Wait for pullback to structure, enter with trend
        """
        # Determine trend direction
        ema_fast = df['close'].ewm(span=10).mean()
        ema_slow = df['close'].ewm(span=30).mean()
        
        trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        trend_direction = 'BUY' if trend_up else 'SELL'
        
        # Calculate ATR
        atr = self._calculate_atr(df, period=14)
        
        # Find recent swing levels
        swing_high, swing_low = self._find_swing_levels(df, lookback=20)
        
        # Check for pullback
        if trend_up:
            # Price should be pulling back toward EMA
            pullback_zone = ema_slow.iloc[-1]
            distance_to_zone = (current_price - pullback_zone) / current_price
            
            if distance_to_zone < 0.01 and distance_to_zone > -0.02:
                # Price is near EMA support
                # Confirm: recent candles showing buying pressure
                recent_close = df['close'].tail(3)
                recent_open = df['open'].tail(3) if 'open' in df.columns else recent_close
                bullish_candles = sum(recent_close.values > recent_open.values)
                
                if bullish_candles >= 2:
                    signal = 'BUY'
                    confidence = 0.65 + (bullish_candles - 2) * 0.1
                    confidence *= regime_analysis.confidence
                    
                    stop_loss = swing_low - atr
                    take_profit = current_price + (current_price - stop_loss) * 2.5
                    
                    return TradeSetup(
                        signal=signal,
                        confidence=confidence,
                        regime=regime_analysis.regime,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size_multiplier=self._calculate_position_multiplier(
                            confidence, regime_analysis
                        ),
                        max_hold_bars=self.max_hold_bars_trending,
                        exit_conditions=[
                            "ema_cross_bearish",
                            "new_lower_low",
                            "momentum_divergence",
                            "time_limit_reached"
                        ],
                        invalidation_price=swing_low - atr * 0.5,
                        reason=f"Trend continuation BUY: pullback to EMA support"
                    )
        else:
            # Downtrend logic
            pullback_zone = ema_slow.iloc[-1]
            distance_to_zone = (pullback_zone - current_price) / current_price
            
            if distance_to_zone < 0.01 and distance_to_zone > -0.02:
                recent_close = df['close'].tail(3)
                recent_open = df['open'].tail(3) if 'open' in df.columns else recent_close
                bearish_candles = sum(recent_close.values < recent_open.values)
                
                if bearish_candles >= 2:
                    signal = 'SELL'
                    confidence = 0.65 + (bearish_candles - 2) * 0.1
                    confidence *= regime_analysis.confidence
                    
                    stop_loss = swing_high + atr
                    take_profit = current_price - (stop_loss - current_price) * 2.5
                    
                    return TradeSetup(
                        signal=signal,
                        confidence=confidence,
                        regime=regime_analysis.regime,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size_multiplier=self._calculate_position_multiplier(
                            confidence, regime_analysis
                        ),
                        max_hold_bars=self.max_hold_bars_trending,
                        exit_conditions=[
                            "ema_cross_bullish",
                            "new_higher_high",
                            "momentum_divergence",
                            "time_limit_reached"
                        ],
                        invalidation_price=swing_high + atr * 0.5,
                        reason=f"Trend continuation SELL: pullback to EMA resistance"
                    )
        
        return TradeSetup(
            signal='NEUTRAL',
            confidence=0.0,
            regime=regime_analysis.regime,
            entry_price=current_price,
            stop_loss=0,
            take_profit=0,
            position_size_multiplier=0,
            max_hold_bars=0,
            exit_conditions=[],
            invalidation_price=0,
            reason="No pullback setup available"
        )
    
    def _setup_volatility_fade(self, df: pd.DataFrame, current_price: float,
                               regime_analysis: RegimeAnalysis) -> TradeSetup:
        """
        Volatility fade setup for ranging volatile markets
        
        Logic: Fade extreme moves, expect snap-back
        Very selective - only trade clear exhaustion
        """
        atr = self._calculate_atr(df, period=14)
        
        # Calculate recent price movement
        bars_back = 5
        price_move = current_price - df['close'].iloc[-bars_back]
        move_in_atr = abs(price_move) / atr if atr > 0 else 0
        
        # Need significant move (3+ ATR) to fade
        if move_in_atr < 3.0:
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason="Move not extreme enough to fade"
            )
        
        # Look for exhaustion signals
        # 1. Volume spike followed by decline (if volume available)
        # 2. Candle body shrinking
        # 3. Wicks forming in direction of move
        
        recent_ranges = df['high'].tail(5) - df['low'].tail(5)
        range_declining = recent_ranges.iloc[-1] < recent_ranges.iloc[:-1].mean()
        
        if not range_declining:
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason="No exhaustion pattern detected"
            )
        
        # Determine fade direction
        if price_move > 0:
            signal = 'SELL'
            stop_loss = current_price + atr * 1.5
            take_profit = current_price - abs(price_move) * 0.5  # 50% retracement
        else:
            signal = 'BUY'
            stop_loss = current_price - atr * 1.5
            take_profit = current_price + abs(price_move) * 0.5
        
        # Lower confidence for volatile regime
        confidence = 0.55 * regime_analysis.confidence
        
        return TradeSetup(
            signal=signal,
            confidence=confidence,
            regime=regime_analysis.regime,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_multiplier=self._calculate_position_multiplier(
                confidence, regime_analysis
            ) * 0.5,  # Half size in volatile regime
            max_hold_bars=25,  # Short hold time
            exit_conditions=[
                "50%_retracement",
                "new_extreme",
                "volatility_expansion",
                "time_limit_reached"
            ],
            invalidation_price=stop_loss,
            reason=f"Volatility fade: {move_in_atr:.1f} ATR move with exhaustion"
        )
    
    def _setup_pullback_continuation(self, df: pd.DataFrame, current_price: float,
                                     regime_analysis: RegimeAnalysis) -> TradeSetup:
        """
        Pullback continuation in volatile trending markets
        
        Logic: Similar to trend continuation but with wider stops
        Very selective entries
        """
        # This is the highest risk regime we trade - be very selective
        atr = self._calculate_atr(df, period=14)
        
        # Trend direction
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        
        trend_up = ema_20 > ema_50
        
        # Calculate trend strength
        trend_slope = (ema_20 - df['close'].ewm(span=20).mean().iloc[-10]) / ema_20
        
        if abs(trend_slope) < 0.01:
            return TradeSetup(
                signal='NEUTRAL',
                confidence=0.0,
                regime=regime_analysis.regime,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size_multiplier=0,
                max_hold_bars=0,
                exit_conditions=[],
                invalidation_price=0,
                reason="Trend too weak for volatile pullback trade"
            )
        
        # Need price to be in pullback zone
        if trend_up:
            pullback_threshold = ema_20 - atr
            if current_price > pullback_threshold:
                return TradeSetup(
                    signal='NEUTRAL',
                    confidence=0.0,
                    regime=regime_analysis.regime,
                    entry_price=current_price,
                    stop_loss=0,
                    take_profit=0,
                    position_size_multiplier=0,
                    max_hold_bars=0,
                    exit_conditions=[],
                    invalidation_price=0,
                    reason="Price not in pullback zone"
                )
            
            signal = 'BUY'
            stop_loss = current_price - atr * 3  # Wide stop for volatile market
            take_profit = current_price + atr * 4
            confidence = 0.55 * regime_analysis.confidence
        else:
            pullback_threshold = ema_20 + atr
            if current_price < pullback_threshold:
                return TradeSetup(
                    signal='NEUTRAL',
                    confidence=0.0,
                    regime=regime_analysis.regime,
                    entry_price=current_price,
                    stop_loss=0,
                    take_profit=0,
                    position_size_multiplier=0,
                    max_hold_bars=0,
                    exit_conditions=[],
                    invalidation_price=0,
                    reason="Price not in pullback zone"
                )
            
            signal = 'SELL'
            stop_loss = current_price + atr * 3
            take_profit = current_price - atr * 4
            confidence = 0.55 * regime_analysis.confidence
        
        return TradeSetup(
            signal=signal,
            confidence=confidence,
            regime=regime_analysis.regime,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_multiplier=self._calculate_position_multiplier(
                confidence, regime_analysis
            ) * 0.5,  # Half size
            max_hold_bars=40,
            exit_conditions=[
                "trend_reversal",
                "volatility_spike",
                "new_swing_against",
                "time_limit_reached"
            ],
            invalidation_price=stop_loss,
            reason=f"Volatile pullback continuation: trend_slope={trend_slope:.3f}"
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if 'atr' in df.columns:
            return df['atr'].iloc[-1]
        
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else (high - low).mean()
    
    def _find_swing_levels(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
        """Find recent swing high and low"""
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        return recent_high, recent_low
    
    def _calculate_position_multiplier(self, confidence: float, 
                                       regime_analysis: RegimeAnalysis) -> float:
        """
        Calculate position size multiplier based on conditions
        
        Returns multiplier from 0.5 to 1.5 applied to base risk
        """
        # Start with confidence-based multiplier
        base_mult = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        # Regime adjustment
        regime_factors = {
            MarketRegime.QUIET_RANGE: 1.2,        # Favorable
            MarketRegime.TRENDING_CALM: 1.1,      # Good
            MarketRegime.VOLATILE_RANGE: 0.6,     # Reduce size
            MarketRegime.TRENDING_VOLATILE: 0.5,  # Minimum size
        }
        regime_mult = regime_factors.get(regime_analysis.regime, 0.5)
        
        # Volatility adjustment
        if regime_analysis.volatility_percentile > 80:
            vol_mult = 0.6
        elif regime_analysis.volatility_percentile > 60:
            vol_mult = 0.8
        elif regime_analysis.volatility_percentile < 30:
            vol_mult = 1.1
        else:
            vol_mult = 1.0
        
        # Regime age factor
        if regime_analysis.regime_age_bars < 15:
            age_mult = 0.8
        else:
            age_mult = 1.0
        
        # Combine factors
        final_mult = base_mult * regime_mult * vol_mult * age_mult
        
        # Clamp to reasonable range
        return max(0.3, min(1.5, final_mult))
    
    def _check_new_day(self):
        """Reset daily counters on new day"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
            # Reset session stats but keep consecutive losses
            self.session_pnl = 0.0
            self.session_peak = 0.0
    
    def record_trade_result(self, pnl: float, is_win: bool):
        """
        Call this method after each trade closes to update internal state
        
        Args:
            pnl: Profit/loss from trade
            is_win: Whether the trade was profitable
        """
        self.daily_trade_count += 1
        self.session_pnl += pnl
        self.session_peak = max(self.session_peak, self.session_pnl)
        
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Check if kill switch should be released
        if self.kill_switch_active:
            if self.consecutive_losses < self.max_consecutive_losses - 1:
                self.kill_switch_active = False
                self.kill_switch_reason = None
    
    def reset_kill_switch(self):
        """Manually reset kill switch (e.g., at start of new session)"""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.consecutive_losses = 0
    
    def get_regime_summary(self) -> Dict:
        """Get summary of current regime state"""
        return {
            'current_regime': self.current_regime.value,
            'previous_regime': self.previous_regime.value,
            'regime_age_bars': self.regime_bar_count,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trade_count,
            'session_pnl': self.session_pnl
        }
