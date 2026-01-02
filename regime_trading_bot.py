"""
Regime-Aware Automated Trading Bot
===================================
Runs continuously, monitors markets, and executes trades on MT5

Features:
- Multi-symbol monitoring
- Automatic trade execution
- Risk management (position sizing, max daily trades, etc.)
- Kill switch protection
- Logging of all activities
"""

import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

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
    """Represents an open position"""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    regime: str
    max_hold_bars: int
    bars_held: int = 0


class RegimeTradingBot:
    """Automated trading bot using the Regime-Aware Strategy"""
    
    def __init__(self, symbols: List[str], risk_per_trade: float = None):
        """
        Initialize the trading bot
        
        Args:
            symbols: List of symbols to trade
            risk_per_trade: Risk per trade as decimal (0.01 = 1%). If None, reads from env or uses default 1%
        """
        self.symbols = symbols
        
        # Load from environment or use defaults
        if risk_per_trade is None:
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.01'))
        self.risk_per_trade = risk_per_trade
        
        # One strategy instance per symbol to maintain state
        self.strategies: Dict[str, RegimeAwareStrategy] = {}
        for symbol in symbols:
            self.strategies[symbol] = RegimeAwareStrategy()
        
        # Open positions tracking
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        
        # Daily stats
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = datetime.now().date()
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '10'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        
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
        # Get credentials from environment
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
        
        # Login to account
        if not mt5.login(login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if not account:
            logger.error("Failed to get account info")
            return False
        
        self.initial_balance = account.balance
        
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
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                stop_loss: float, multiplier: float = 1.0) -> float:
        """
        Calculate position size based on risk
        
        Returns volume in lots
        """
        account = mt5.account_info()
        if not account:
            return 0.01  # Minimum lot
        
        # Risk amount
        risk_amount = account.balance * self.risk_per_trade * multiplier
        
        # Get symbol info for lot size
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        # Calculate pip value and lot size
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0.01
        
        # Simple calculation (synthetic indices have different pip values)
        # For synthetics, 1 lot typically = $1 per point
        contract_size = symbol_info.trade_contract_size
        
        # Position size based on risk
        volume = risk_amount / (stop_distance * contract_size) if contract_size > 0 else 0.01
        
        # Clamp to valid range
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        volume = max(min_lot, min(max_lot, volume))
        volume = round(volume / lot_step) * lot_step
        
        return max(min_lot, volume)
    
    def open_trade(self, symbol: str, direction: str, volume: float,
                   stop_loss: float, take_profit: float, regime: str,
                   max_hold_bars: int) -> Optional[int]:
        """
        Open a trade on MT5
        
        Returns ticket number or None if failed
        """
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick for {symbol}")
            return None
        
        # Determine order type and price
        if direction == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Use FOK filling mode (Fill or Kill) - works with Deriv
        type_filling = mt5.ORDER_FILLING_FOK
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,  # Slippage tolerance
            "magic": 123456,  # Magic number to identify our trades
            "comment": f"Regime:{regime}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
        
        # Track position
        self.positions[symbol] = Position(
            ticket=result.order,
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            regime=regime,
            max_hold_bars=max_hold_bars
        )
        
        self.daily_trades += 1
        
        logger.info(f"✅ OPENED: {direction} {volume} {symbol} @ {price:.5f}")
        logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        logger.info(f"   Ticket: {result.order}")
        
        return result.order
    
    def close_trade(self, symbol: str, reason: str = "manual") -> bool:
        """Close an open trade"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        if position.direction == 'BUY':
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Close:{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close {symbol}: {result}")
            return False
        
        # Calculate P&L
        if position.direction == 'BUY':
            pnl_points = price - position.entry_price
        else:
            pnl_points = position.entry_price - price
        
        pnl_percent = (pnl_points / position.entry_price) * 100
        
        emoji = "✅" if pnl_points > 0 else "❌"
        logger.info(f"{emoji} CLOSED: {symbol} | P&L: {pnl_percent:.3f}% | Reason: {reason}")
        
        # Update strategy
        self.strategies[symbol].record_trade_result(pnl_points, is_win=pnl_points > 0)
        
        # Remove from tracking
        del self.positions[symbol]
        
        return True
    
    def check_open_positions(self):
        """Check and manage open positions"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            position.bars_held += 1
            
            # Check time-based exit
            if position.bars_held >= position.max_hold_bars:
                logger.info(f"Time exit for {symbol}: {position.bars_held} bars held")
                self.close_trade(symbol, reason="time_exit")
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits are exceeded"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            
            # Reset kill switches on new day
            for strategy in self.strategies.values():
                strategy.reset_kill_switch()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Check daily loss limit
        account = mt5.account_info()
        if account:
            daily_return = (account.balance - self.initial_balance) / self.initial_balance
            if daily_return < -self.max_daily_loss:
                logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
                return False
        
        return True
    
    def analyze_and_trade(self, symbol: str):
        """Analyze a symbol and trade if signal found"""
        # Skip if already in position
        if symbol in self.positions:
            return
        
        # Get strategy
        strategy = self.strategies[symbol]
        
        # Get data
        df = self.get_ohlc_data(symbol)
        if df is None or len(df) < 150:
            return
        
        current_price = df['close'].iloc[-1]
        
        # Analyze
        signal, confidence, metadata = strategy.analyze(df, current_price)
        
        # Log if interesting
        regime = metadata.get('regime', 'unknown')
        if signal != 'NEUTRAL':
            logger.info(f"{symbol}: {signal} signal ({confidence:.0%}) - {regime}")
            logger.info(f"   Reason: {metadata.get('reason', 'N/A')}")
        
        # Check for trade
        if signal in ['BUY', 'SELL'] and confidence >= 0.50:
            # Get trade parameters
            stop_loss = metadata.get('stop_loss')
            take_profit = metadata.get('take_profit')
            multiplier = metadata.get('position_size_multiplier', 1.0)
            max_hold = metadata.get('max_hold_bars', 50)
            
            if stop_loss and take_profit:
                # Calculate position size
                volume = self.calculate_position_size(
                    symbol, current_price, stop_loss, multiplier
                )
                
                # Open trade
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
        
        print("\n" + "="*70)
        print(f"REGIME TRADING BOT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        if account:
            pnl = account.balance - self.initial_balance
            pnl_pct = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            emoji = "📈" if pnl >= 0 else "📉"
            
            print(f"💰 Balance: ${account.balance:,.2f} | {emoji} P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        
        print(f"📊 Daily trades: {self.daily_trades}/{self.max_daily_trades}")
        print(f"📈 Open positions: {len(self.positions)}")
        
        # Print regime status for each symbol
        print("\n" + "-"*70)
        print(f"{'Symbol':<25} {'Regime':<18} {'Age':>4} {'Vol%':>5} {'Position':>10}")
        print("-"*70)
        
        for symbol in self.symbols:
            strategy = self.strategies[symbol]
            summary = strategy.get_regime_summary()
            
            # Get latest volatility percentile
            df = self.get_ohlc_data(symbol, bars=150)
            vol_pct = ""
            if df is not None:
                from utils.regime_detector import calculate_volatility_metrics
                vol_metrics = calculate_volatility_metrics(df)
                if vol_metrics['is_valid']:
                    vol_pct = f"{vol_metrics['volatility_percentile']:.0f}%"
            
            pos_str = ""
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos_str = f"{pos.direction} {pos.volume:.2f}"
            
            print(f"{symbol:<25} {summary['current_regime']:<18} "
                  f"{summary['regime_age_bars']:>4} {vol_pct:>5} {pos_str:>10}")
        
        print("-"*70)
        
        # Show open positions details
        if self.positions:
            print("\n📋 OPEN POSITIONS:")
            for symbol, pos in self.positions.items():
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current = tick.bid if pos.direction == 'BUY' else tick.ask
                    pnl = (current - pos.entry_price) if pos.direction == 'BUY' else (pos.entry_price - current)
                    pnl_pct = (pnl / pos.entry_price) * 100
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    print(f"   {emoji} {pos.direction} {symbol}: {pnl_pct:+.3f}% | Bars: {pos.bars_held}/{pos.max_hold_bars}")
        
        print("="*70)
    
    def run(self, interval_seconds: int = 60):
        """
        Main bot loop
        
        Args:
            interval_seconds: Time between analysis cycles
        """
        if not self.connect():
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Bot started - monitoring {len(self.symbols)} symbols")
        logger.info(f"Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"Max daily trades: {self.max_daily_trades}")
        
        print("\n" + "="*70)
        print("🤖 REGIME TRADING BOT STARTED")
        print("="*70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Risk per trade: {self.risk_per_trade:.1%}")
        print(f"Analysis interval: {interval_seconds} seconds")
        print("\nPress Ctrl+C to stop")
        print("="*70)
        
        try:
            iteration = 0
            while self.is_running:
                iteration += 1
                
                # Check daily limits
                if not self.check_daily_limits():
                    logger.warning("Daily limits reached - pausing trading")
                    time.sleep(60)
                    continue
                
                # Check open positions
                self.check_open_positions()
                
                # Analyze each symbol
                for symbol in self.symbols:
                    try:
                        self.analyze_and_trade(symbol)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Print status every 5 iterations
                if iteration % 5 == 0:
                    self.print_status()
                
                # Wait
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Stopping bot (user interrupt)")
        finally:
            self.is_running = False
            
            # Close all positions on shutdown (optional)
            # for symbol in list(self.positions.keys()):
            #     self.close_trade(symbol, reason="bot_shutdown")
            
            self.print_status()
            self.disconnect()
            
            logger.info("Bot stopped")


def main():
    """Start the trading bot"""
    # Symbols to trade (Volatility indices tend to have the best regimes)
    symbols = [
        'Volatility 50 Index',
        'Volatility 75 Index',
        'Volatility 100 Index',
        'Boom 1000 Index',
        'Crash 1000 Index',
    ]
    
    # Create and run bot
    bot = RegimeTradingBot(
        symbols=symbols,
        risk_per_trade=0.01  # 1% risk per trade
    )
    
    # Run with 60-second intervals (5-minute bars)
    bot.run(interval_seconds=60)


if __name__ == "__main__":
    main()
