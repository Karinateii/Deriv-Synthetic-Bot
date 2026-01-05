"""
Regime-Aware Automated Trading Bot v2.0
========================================
Professional-grade trading bot with advanced money management

Features:
- Multi-symbol monitoring
- Dynamic volatility-adjusted position sizing
- Partial profit taking (scale out at 1R, 2R)
- Trailing stops for trend trades
- Max concurrent positions limit
- Consecutive loss protection
- Regime-based trade management
- Equity curve management
- Kill switch protection
- Comprehensive logging
"""

import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 not installed. Run: pip install MetaTrader5")
    sys.exit(1)

import pandas as pd
import numpy as np
from loguru import logger

from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime
from config import Config


@dataclass
class Position:
    """Represents an open position with advanced tracking"""
    ticket: int
    symbol: str
    direction: str
    initial_volume: float  # Original volume
    current_volume: float  # Volume after partial closes
    entry_price: float
    stop_loss: float
    original_stop_loss: float  # Keep original for R calculation
    take_profit: float
    entry_time: datetime
    regime: str
    max_hold_bars: int
    bars_held: int = 0
    highest_price: float = 0.0  # For trailing stop (buys)
    lowest_price: float = float('inf')  # For trailing stop (sells)
    partial_1r_taken: bool = False  # Took profit at 1R
    partial_2r_taken: bool = False  # Took profit at 2R
    trailing_activated: bool = False  # Trailing stop is active
    
    def __post_init__(self):
        """Initialize tracking prices"""
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == float('inf'):
            self.lowest_price = self.entry_price
    
    @property
    def risk_distance(self) -> float:
        """1R = distance from entry to original stop"""
        return abs(self.entry_price - self.original_stop_loss)
    
    def current_r_multiple(self, current_price: float) -> float:
        """Calculate current R-multiple (profit in units of risk)"""
        if self.risk_distance == 0:
            return 0
        if self.direction == 'BUY':
            return (current_price - self.entry_price) / self.risk_distance
        else:
            return (self.entry_price - current_price) / self.risk_distance


class RegimeTradingBot:
    """Professional automated trading bot using the Regime-Aware Strategy"""
    
    # === CONFIGURATION CONSTANTS ===
    MAX_CONCURRENT_POSITIONS = 3  # Never have more than this open
    MAX_CONSECUTIVE_LOSSES = 3    # Pause after this many losses in a row
    
    # Partial profit taking percentages
    PARTIAL_1R_PERCENT = 0.30     # Take 30% at 1R
    PARTIAL_2R_PERCENT = 0.30     # Take 30% at 2R (leaving 40% to run)
    
    # Trailing stop settings
    TRAILING_ACTIVATION_R = 1.5   # Activate trailing after 1.5R profit
    TRAILING_DISTANCE_ATR = 2.0   # Trail by 2x ATR
    
    # Volatility adjustment for position sizing
    VOL_ADJUSTMENT_BASELINE = 50  # Baseline volatility percentile
    MIN_VOL_MULTIPLIER = 0.5      # Minimum position size multiplier
    MAX_VOL_MULTIPLIER = 1.5      # Maximum position size multiplier
    
    # Equity curve management
    DRAWDOWN_REDUCE_THRESHOLD = 0.10  # Reduce size after 10% drawdown
    DRAWDOWN_REDUCE_FACTOR = 0.50     # Cut position size by 50%
    PROFIT_INCREASE_THRESHOLD = 0.25  # Increase size after 25% profit
    PROFIT_INCREASE_FACTOR = 1.25     # Increase position size by 25%
    
    def __init__(self, symbols: List[str], risk_per_trade: float = None):
        """
        Initialize the trading bot
        
        Args:
            symbols: List of symbols to trade
            risk_per_trade: Base risk per trade as decimal (0.01 = 1%)
        """
        self.symbols = symbols
        
        # Load from environment or use defaults
        if risk_per_trade is None:
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.01'))
        self.base_risk_per_trade = risk_per_trade
        self.current_risk_per_trade = risk_per_trade  # Adjusted based on equity curve
        
        # One strategy instance per symbol to maintain state
        self.strategies: Dict[str, RegimeAwareStrategy] = {}
        for symbol in symbols:
            self.strategies[symbol] = RegimeAwareStrategy()
        
        # Open positions tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        
        # Performance tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = datetime.now().date()
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '10'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        
        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.trade_results: List[bool] = []  # Recent trade results (True=win, False=loss)
        
        # Equity curve tracking
        self.peak_balance = 0.0
        self.equity_adjustment_factor = 1.0
        
        # ATR cache for trailing stops (refreshed each cycle)
        self.atr_cache: Dict[str, float] = {}
        
        # Bot state
        self.is_running = False
        self.start_time = None
        self.initial_balance = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"regime_bot_{datetime.now().strftime('%Y%m%d')}.log"
        
        logger.add(
            log_file,
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def connect(self) -> bool:
        """Connect to MT5 using environment variables"""
        login = os.getenv('MT5_LOGIN')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER', 'Deriv-Demo')
        
        if not login or not password:
            logger.error("MT5_LOGIN and MT5_PASSWORD must be set in environment variables")
            return False
        
        try:
            login = int(login)
        except ValueError:
            logger.error("MT5_LOGIN must be a number")
            return False
        
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        if not mt5.login(login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if not account:
            logger.error("Failed to get account info")
            return False
        
        self.initial_balance = account.balance
        self.peak_balance = account.balance
        
        logger.info(f"Connected to MT5")
        logger.info(f"Account: {account.login}")
        logger.info(f"Server: {account.server}")
        logger.info(f"Balance: ${account.balance:.2f}")
        logger.info(f"Leverage: 1:{account.leverage}")
        
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_ohlc_data(self, symbol: str, bars: int = 200) -> Optional[pd.DataFrame]:
        """Get OHLC data for a symbol"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def get_volatility_percentile(self, symbol: str) -> float:
        """Get current volatility percentile for a symbol"""
        df = self.get_ohlc_data(symbol, bars=150)
        if df is None:
            return 50.0  # Default to baseline
        
        try:
            from utils.regime_detector import calculate_volatility_metrics
            vol_metrics = calculate_volatility_metrics(df)
            if vol_metrics['is_valid']:
                return vol_metrics['volatility_percentile']
        except Exception:
            pass
        
        return 50.0
    
    def update_equity_curve_adjustment(self):
        """Adjust position sizing based on equity curve"""
        account = mt5.account_info()
        if not account:
            return
        
        current_balance = account.balance
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown from peak
        drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        # Calculate profit from initial
        profit_pct = (current_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Adjust position sizing
        if drawdown >= self.DRAWDOWN_REDUCE_THRESHOLD:
            self.equity_adjustment_factor = self.DRAWDOWN_REDUCE_FACTOR
            logger.warning(f"📉 Drawdown {drawdown:.1%} - reducing position sizes to {self.equity_adjustment_factor:.0%}")
        elif profit_pct >= self.PROFIT_INCREASE_THRESHOLD:
            self.equity_adjustment_factor = self.PROFIT_INCREASE_FACTOR
            logger.info(f"📈 Profit {profit_pct:.1%} - increasing position sizes to {self.equity_adjustment_factor:.0%}")
        else:
            self.equity_adjustment_factor = 1.0
        
        # Update current risk
        self.current_risk_per_trade = self.base_risk_per_trade * self.equity_adjustment_factor
    
    def calculate_volatility_adjusted_risk(self, symbol: str) -> float:
        """Calculate risk adjusted for current volatility"""
        vol_pct = self.get_volatility_percentile(symbol)
        
        # Adjust risk based on volatility
        # Low vol (20%) = lower risk, High vol (80%) = higher risk
        vol_multiplier = vol_pct / self.VOL_ADJUSTMENT_BASELINE
        vol_multiplier = max(self.MIN_VOL_MULTIPLIER, min(self.MAX_VOL_MULTIPLIER, vol_multiplier))
        
        adjusted_risk = self.current_risk_per_trade * vol_multiplier
        
        return adjusted_risk
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                stop_loss: float, regime_multiplier: float = 1.0) -> float:
        """
        Calculate position size with volatility and equity adjustments
        
        Returns volume in lots
        """
        account = mt5.account_info()
        if not account:
            return 0.01
        
        # Get volatility-adjusted risk
        adjusted_risk = self.calculate_volatility_adjusted_risk(symbol)
        
        # Apply regime multiplier and calculate risk amount
        risk_amount = account.balance * adjusted_risk * regime_multiplier
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return 0.01
        
        # Position size based on risk
        contract_size = symbol_info.trade_contract_size
        volume = risk_amount / (stop_distance * contract_size) if contract_size > 0 else 0.01
        
        # Clamp to valid range
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        volume = max(min_lot, min(max_lot, volume))
        volume = round(volume / lot_step) * lot_step
        
        return max(min_lot, volume)
    
    def can_open_new_position(self) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        # Check max concurrent positions
        if len(self.positions) >= self.MAX_CONCURRENT_POSITIONS:
            return False, f"Max {self.MAX_CONCURRENT_POSITIONS} positions reached"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            return False, f"Paused after {self.consecutive_losses} consecutive losses"
        
        # Check daily limits
        if self.daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Check daily loss limit
        account = mt5.account_info()
        if account:
            daily_return = (account.balance - self.initial_balance) / self.initial_balance
            if daily_return < -self.max_daily_loss:
                return False, f"Daily loss limit hit: {daily_return:.2%}"
        
        return True, "OK"
    
    def open_trade(self, symbol: str, direction: str, volume: float,
                   stop_loss: float, take_profit: float, regime: str,
                   max_hold_bars: int) -> Optional[int]:
        """Open a trade on MT5"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        if direction == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Regime:{regime}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        # Track position with enhanced data
        self.positions[symbol] = Position(
            ticket=result.order,
            symbol=symbol,
            direction=direction,
            initial_volume=volume,
            current_volume=volume,
            entry_price=price,
            stop_loss=stop_loss,
            original_stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            regime=regime,
            max_hold_bars=max_hold_bars,
            highest_price=price,
            lowest_price=price
        )
        
        self.daily_trades += 1
        
        logger.info(f"✅ OPENED: {direction} {volume:.2f} {symbol} @ {price:.5f}")
        logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Risk: {self.current_risk_per_trade:.2%}")
        logger.info(f"   Ticket: {result.order} | Open positions: {len(self.positions)}/{self.MAX_CONCURRENT_POSITIONS}")
        
        return result.order
    
    def close_partial(self, symbol: str, percent: float, reason: str) -> bool:
        """Close a percentage of the position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Calculate volume to close
        close_volume = position.current_volume * percent
        
        # Get symbol info for lot step
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False
        
        # Round to valid lot size
        lot_step = symbol_info.volume_step
        min_lot = symbol_info.volume_min
        close_volume = round(close_volume / lot_step) * lot_step
        
        # Ensure we're closing at least min lot and leaving at least min lot
        remaining = position.current_volume - close_volume
        if close_volume < min_lot or remaining < min_lot:
            # Can't do partial, skip
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        if position.direction == 'BUY':
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Partial:{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed partial close {symbol}: {result}")
            return False
        
        # Update position volume
        position.current_volume -= close_volume
        
        # Calculate partial P&L
        if position.direction == 'BUY':
            pnl_r = (price - position.entry_price) / position.risk_distance
        else:
            pnl_r = (position.entry_price - price) / position.risk_distance
        
        logger.info(f"💰 PARTIAL CLOSE: {close_volume:.2f} {symbol} @ {price:.5f} | {pnl_r:.1f}R | {reason}")
        logger.info(f"   Remaining: {position.current_volume:.2f} lots")
        
        return True
    
    def close_trade(self, symbol: str, reason: str = "manual") -> bool:
        """Close entire position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        if position.direction == 'BUY':
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.current_volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Close:{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close {symbol}: {result}")
            return False
        
        # Calculate final P&L
        r_multiple = position.current_r_multiple(price)
        is_win = r_multiple > 0
        
        # Update consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Track result
        self.trade_results.append(is_win)
        if len(self.trade_results) > 20:
            self.trade_results.pop(0)
        
        emoji = "✅" if is_win else "❌"
        logger.info(f"{emoji} CLOSED: {symbol} | {r_multiple:+.2f}R | {reason}")
        logger.info(f"   Consecutive losses: {self.consecutive_losses}")
        
        # Update strategy
        pnl_points = price - position.entry_price if position.direction == 'BUY' else position.entry_price - price
        self.strategies[symbol].record_trade_result(pnl_points, is_win=is_win)
        
        del self.positions[symbol]
        return True
    
    def update_trailing_stop(self, symbol: str, position: Position, current_price: float):
        """Update trailing stop for a position"""
        # Get ATR for this symbol
        atr = self.atr_cache.get(symbol, 0)
        if atr == 0:
            return
        
        trailing_distance = atr * self.TRAILING_DISTANCE_ATR
        
        if position.direction == 'BUY':
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
            
            # Calculate new stop
            new_stop = position.highest_price - trailing_distance
            
            # Only move stop up, never down
            if new_stop > position.stop_loss and new_stop < current_price:
                # Modify the stop loss in MT5
                if self._modify_stop_loss(symbol, position, new_stop):
                    position.stop_loss = new_stop
                    logger.info(f"📈 Trailing stop {symbol}: SL moved to {new_stop:.5f}")
        else:
            # Update lowest price
            if current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Calculate new stop
            new_stop = position.lowest_price + trailing_distance
            
            # Only move stop down, never up
            if new_stop < position.stop_loss and new_stop > current_price:
                if self._modify_stop_loss(symbol, position, new_stop):
                    position.stop_loss = new_stop
                    logger.info(f"📉 Trailing stop {symbol}: SL moved to {new_stop:.5f}")
    
    def _modify_stop_loss(self, symbol: str, position: Position, new_stop: float) -> bool:
        """Modify stop loss of an existing position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": new_stop,
            "tp": position.take_profit,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.debug(f"Failed to modify SL for {symbol}: {result}")
            return False
        
        return True
    
    def manage_open_positions(self):
        """Check and manage open positions - partial profits, trailing stops, regime changes"""
        # Sync with MT5 first
        mt5_positions = mt5.positions_get()
        mt5_tickets = set()
        if mt5_positions:
            mt5_tickets = {p.ticket for p in mt5_positions}
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            # Check if position was closed externally
            if position.ticket not in mt5_tickets:
                logger.info(f"🔄 Position {symbol} closed externally (SL/TP hit or manual)")
                # We don't know the result, assume loss for safety
                self.consecutive_losses += 1
                self.strategies[symbol].record_trade_result(0, is_win=False)
                del self.positions[symbol]
                continue
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue
            
            current_price = tick.bid if position.direction == 'BUY' else tick.ask
            r_multiple = position.current_r_multiple(current_price)
            
            position.bars_held += 1
            
            # === PARTIAL PROFIT TAKING ===
            
            # Take 30% at 1R
            if not position.partial_1r_taken and r_multiple >= 1.0:
                if self.close_partial(symbol, self.PARTIAL_1R_PERCENT, "1R_profit"):
                    position.partial_1r_taken = True
                    # Move stop to breakeven
                    if self._modify_stop_loss(symbol, position, position.entry_price):
                        position.stop_loss = position.entry_price
                        logger.info(f"🔒 Breakeven stop set for {symbol}")
            
            # Take 30% at 2R
            if not position.partial_2r_taken and r_multiple >= 2.0:
                if self.close_partial(symbol, self.PARTIAL_2R_PERCENT, "2R_profit"):
                    position.partial_2r_taken = True
            
            # === TRAILING STOP ===
            
            # Activate trailing after 1.5R
            if r_multiple >= self.TRAILING_ACTIVATION_R:
                if not position.trailing_activated:
                    position.trailing_activated = True
                    logger.info(f"🎯 Trailing stop activated for {symbol} at {r_multiple:.1f}R")
                
                self.update_trailing_stop(symbol, position, current_price)
            
            # === REGIME CHANGE EXIT ===
            
            # Check if regime has changed unfavorably
            strategy = self.strategies[symbol]
            summary = strategy.get_regime_summary()
            current_regime = summary.get('current_regime', 'unknown')
            
            # Exit if regime turns chaotic or unfavorable
            if current_regime in ['chaotic', 'transition']:
                if r_multiple > 0:  # Only exit if in profit
                    logger.info(f"⚠️ Regime change to {current_regime} - closing {symbol}")
                    self.close_trade(symbol, reason=f"regime_change_{current_regime}")
                    continue
            
            # === TIME-BASED EXIT ===
            
            if position.bars_held >= position.max_hold_bars:
                logger.info(f"⏰ Time exit for {symbol}: {position.bars_held} bars held")
                self.close_trade(symbol, reason="time_exit")
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits are exceeded"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            # New day - reset counters
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            self.consecutive_losses = 0  # Reset on new day
            
            # Reset kill switches
            for strategy in self.strategies.values():
                strategy.reset_kill_switch()
            
            logger.info("🌅 New trading day - counters reset")
        
        can_trade, reason = self.can_open_new_position()
        return can_trade
    
    def analyze_and_trade(self, symbol: str):
        """Analyze a symbol and trade if signal found"""
        # Skip if already in position for this symbol
        if symbol in self.positions:
            return
        
        # Check if we can open new positions
        can_trade, reason = self.can_open_new_position()
        if not can_trade:
            return
        
        strategy = self.strategies[symbol]
        
        # Get data and update ATR cache
        df = self.get_ohlc_data(symbol)
        if df is None or len(df) < 150:
            return
        
        # Cache ATR for trailing stops
        self.atr_cache[symbol] = self.calculate_atr(df)
        
        current_price = df['close'].iloc[-1]
        
        # Analyze
        signal, confidence, metadata = strategy.analyze(df, current_price)
        
        regime = metadata.get('regime', 'unknown')
        
        # Log non-neutral signals
        if signal != 'NEUTRAL':
            logger.info(f"{symbol}: {signal} signal ({confidence:.0%}) - {regime}")
            logger.info(f"   Reason: {metadata.get('reason', 'N/A')}")
        
        # === ENTRY CONFIRMATION ===
        # Only trade in favorable regimes with sufficient confidence
        if signal in ['BUY', 'SELL'] and confidence >= Config.REGIME_MIN_CONFIDENCE:
            
            # Skip chaotic or transition regimes
            if regime in ['chaotic', 'transition']:
                logger.info(f"   ⏭️ Skipping - unfavorable regime: {regime}")
                return
            
            stop_loss = metadata.get('stop_loss')
            take_profit = metadata.get('take_profit')
            multiplier = metadata.get('position_size_multiplier', 1.0)
            max_hold = metadata.get('max_hold_bars', 50)
            
            if stop_loss and take_profit:
                # Calculate R:R ratio
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
                
                # Skip if R:R is too low
                if rr_ratio < 1.5:
                    logger.info(f"   ⏭️ Skipping - R:R too low: {rr_ratio:.2f}")
                    return
                
                # Calculate position size with all adjustments
                volume = self.calculate_position_size(
                    symbol, current_price, stop_loss, multiplier
                )
                
                self.open_trade(
                    symbol=symbol,
                    direction=signal,
                    volume=volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    regime=regime,
                    max_hold_bars=max_hold
                )
    
    def print_status(self):
        """Print current status"""
        account = mt5.account_info()
        
        print("\n" + "="*80)
        print(f"REGIME TRADING BOT v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if account:
            pnl = account.balance - self.initial_balance
            pnl_pct = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            drawdown = (self.peak_balance - account.balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
            emoji = "📈" if pnl >= 0 else "📉"
            
            print(f"💰 Balance: ${account.balance:,.2f} | {emoji} P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            print(f"📊 Drawdown: {drawdown:.2f}% | Peak: ${self.peak_balance:,.2f}")
        
        print(f"📈 Positions: {len(self.positions)}/{self.MAX_CONCURRENT_POSITIONS} | Daily trades: {self.daily_trades}/{self.max_daily_trades}")
        print(f"⚠️ Consecutive losses: {self.consecutive_losses}/{self.MAX_CONSECUTIVE_LOSSES} | Risk: {self.current_risk_per_trade:.2%}")
        
        # Win rate
        if self.trade_results:
            win_rate = sum(self.trade_results) / len(self.trade_results) * 100
            print(f"🎯 Recent win rate: {win_rate:.0f}% ({sum(self.trade_results)}/{len(self.trade_results)})")
        
        print("\n" + "-"*80)
        print(f"{'Symbol':<25} {'Regime':<18} {'Age':>4} {'Vol%':>5} {'Position':>12} {'R':>6}")
        print("-"*80)
        
        for symbol in self.symbols:
            strategy = self.strategies[symbol]
            summary = strategy.get_regime_summary()
            
            vol_pct = f"{self.get_volatility_percentile(symbol):.0f}%"
            
            pos_str = ""
            r_str = ""
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos_str = f"{pos.direction} {pos.current_volume:.2f}"
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current = tick.bid if pos.direction == 'BUY' else tick.ask
                    r_mult = pos.current_r_multiple(current)
                    r_str = f"{r_mult:+.1f}R"
            
            print(f"{symbol:<25} {summary['current_regime']:<18} "
                  f"{summary['regime_age_bars']:>4} {vol_pct:>5} {pos_str:>12} {r_str:>6}")
        
        print("-"*80)
        
        # Show open positions details
        if self.positions:
            print("\n📋 OPEN POSITIONS:")
            for symbol, pos in self.positions.items():
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current = tick.bid if pos.direction == 'BUY' else tick.ask
                    r_mult = pos.current_r_multiple(current)
                    emoji = "🟢" if r_mult >= 0 else "🔴"
                    
                    status = []
                    if pos.partial_1r_taken:
                        status.append("1R✓")
                    if pos.partial_2r_taken:
                        status.append("2R✓")
                    if pos.trailing_activated:
                        status.append("Trail")
                    status_str = " | ".join(status) if status else "Holding"
                    
                    print(f"   {emoji} {pos.direction} {symbol}: {r_mult:+.2f}R | "
                          f"Bars: {pos.bars_held}/{pos.max_hold_bars} | {status_str}")
        
        print("="*80)
    
    def run(self, interval_seconds: int = 60):
        """Main bot loop"""
        if not self.connect():
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Bot started - monitoring {len(self.symbols)} symbols")
        logger.info(f"Base risk per trade: {self.base_risk_per_trade:.1%}")
        logger.info(f"Max concurrent positions: {self.MAX_CONCURRENT_POSITIONS}")
        logger.info(f"Max consecutive losses: {self.MAX_CONSECUTIVE_LOSSES}")
        logger.info(f"Max daily trades: {self.max_daily_trades}")
        
        print("\n" + "="*80)
        print("🤖 REGIME TRADING BOT v2.0 STARTED")
        print("="*80)
        print(f"Symbols: {len(self.symbols)} instruments")
        print(f"Base risk per trade: {self.base_risk_per_trade:.1%}")
        print(f"Max positions: {self.MAX_CONCURRENT_POSITIONS}")
        print(f"Partial profit: {self.PARTIAL_1R_PERCENT:.0%} at 1R, {self.PARTIAL_2R_PERCENT:.0%} at 2R")
        print(f"Trailing stop: Activates at {self.TRAILING_ACTIVATION_R}R, trails by {self.TRAILING_DISTANCE_ATR}x ATR")
        print(f"Analysis interval: {interval_seconds} seconds")
        print("\nPress Ctrl+C to stop")
        print("="*80)
        
        try:
            iteration = 0
            while self.is_running:
                iteration += 1
                
                # Update equity curve adjustments
                self.update_equity_curve_adjustment()
                
                # Check daily limits
                if not self.check_daily_limits():
                    logger.info("Trading paused - limits reached")
                    time.sleep(60)
                    continue
                
                # Manage existing positions (partials, trailing, regime changes)
                self.manage_open_positions()
                
                # Analyze each symbol for new entries
                for symbol in self.symbols:
                    try:
                        self.analyze_and_trade(symbol)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Print status every 5 iterations
                if iteration % 5 == 0:
                    self.print_status()
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Stopping bot (user interrupt)")
        finally:
            self.is_running = False
            self.print_status()
            self.disconnect()
            logger.info("Bot stopped")


def main():
    """Start the trading bot"""
    symbols = [
        'Volatility 10 (1s) Index',
        'Volatility 10 Index',
        'Volatility 15 (1s) Index',
        'Volatility 25 Index',
        'Volatility 30 (1s) Index',
        'Volatility 50 (1s) Index',
        'Volatility 50 Index',
        'Volatility 75 Index',
        'Volatility 100 Index',
        'Boom 1000 Index',
        'Crash 1000 Index',
    ]
    
    bot = RegimeTradingBot(
        symbols=symbols,
        risk_per_trade=0.01  # 1% base risk per trade
    )
    
    bot.run(interval_seconds=60)


if __name__ == "__main__":
    main()
