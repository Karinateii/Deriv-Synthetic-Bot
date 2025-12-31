"""
MT5 Trading Bot with Regime-Aware Strategy
Designed for synthetic indices with adaptive regime detection
"""
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from loguru import logger

from config import Config
from api.mt5_api import MT5API
from api.mt5_market_data import MT5MarketDataHandler
from risk.risk_manager import RiskManager
from utils.indicators import TechnicalIndicators
from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime, TradabilityState

# Configure logger
logger.remove()
logger.add(sys.stderr, level=Config.LOG_LEVEL)
logger.add(
    Config.LOG_DIR / f"trading_regime_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="1 day",
    retention="30 days",
    level=Config.LOG_LEVEL
)


class DerivBotRegime:
    """MT5-based trading bot with regime-aware strategy"""
    
    def __init__(self):
        self.mt5: MT5API = None
        self.market_data: dict = {}
        self.strategy: RegimeAwareStrategy = None
        self.risk_manager: RiskManager = None
        self.is_running = False
        
        # Trade tracking
        self.open_positions = {}
        self.trade_history = []
        
        logger.info("=" * 60)
        logger.info("DerivBot - Regime-Aware Strategy")
        logger.info("Capital Preservation Mode")
        logger.info("=" * 60)
        
    def initialize(self):
        """Initialize bot components"""
        try:
            # Connect to MT5
            logger.info("Connecting to MT5...")
            self.mt5 = MT5API(
                login=Config.MT5_LOGIN,
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER
            )
            
            if not self.mt5.connect():
                raise Exception("Failed to connect to MT5")
            
            # Get account balance
            balance = self.mt5.get_account_balance()
            account = self.mt5.get_account_info()
            logger.success(f"Account: {account.get('login', 'N/A')} | Balance: ${balance:.2f}")
            
            # Initialize risk manager
            trading_capital = Config.INITIAL_CAPITAL if Config.INITIAL_CAPITAL > 0 else balance
            self.risk_manager = RiskManager(trading_capital)
            logger.info(f"Risk manager initialized with ${trading_capital:.2f} capital")
            
            # Initialize regime-aware strategy
            logger.info("Initializing Regime-Aware Strategy...")
            self.strategy = RegimeAwareStrategy()
            
            # Log strategy parameters
            logger.info(f"  Min confidence: {self.strategy.min_confidence:.0%}")
            logger.info(f"  Base risk: {self.strategy.base_risk_percent:.1%}")
            logger.info(f"  Max daily trades: {self.strategy.max_daily_trades}")
            logger.info(f"  Kill switch: consecutive losses > {self.strategy.max_consecutive_losses}")
            
            logger.success("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def start_data_collection(self, symbols: list):
        """Start collecting market data for symbols"""
        for symbol in symbols:
            try:
                logger.info(f"Starting data collection for {symbol}")
                handler = MT5MarketDataHandler(self.mt5, symbol)
                handler.start()
                
                self.market_data[symbol] = handler
                logger.success(f"Data collection started for {symbol}")
                
            except Exception as e:
                logger.error(f"Error starting data collection for {symbol}: {e}")
    
    def get_regime_status(self):
        """Get current regime status summary"""
        summary = self.strategy.get_regime_summary()
        return summary
    
    def analyze_and_trade(self, symbol: str):
        """Analyze symbol and execute trades using regime-aware strategy"""
        try:
            handler = self.market_data.get(symbol)
            if not handler:
                return
            
            # Update with latest data
            handler.update()
            
            # Get data
            df = handler.get_dataframe()
            
            if len(df) < 200:  # Need enough data for regime analysis
                logger.debug(f"{symbol}: Waiting for data ({len(df)}/200 bars)")
                return
            
            # Add technical indicators
            df = TechnicalIndicators.calculate_all(df)
            
            current_price = df.iloc[-1]['close']
            
            # Get strategy signal
            signal, confidence, metadata = self.strategy.analyze(df, current_price)
            
            # Log regime info
            regime = metadata.get('regime', 'unknown')
            tradability = metadata.get('tradability', 'N/A')
            reason = metadata.get('reason', 'N/A')
            
            # Format log message based on signal
            if signal == 'NEUTRAL':
                logger.info(
                    f"{symbol} | Regime: {regime} | Signal: {signal} | "
                    f"Reason: {reason[:50]}..."
                )
            else:
                logger.info(
                    f"{symbol} | Regime: {regime} | Signal: {signal} | "
                    f"Confidence: {confidence:.1%} | Price: {current_price:.5f}"
                )
            
            # Execute trade if we have a valid signal
            if signal in ['BUY', 'SELL'] and confidence >= self.strategy.min_confidence:
                self._execute_trade(symbol, signal, confidence, current_price, metadata)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_trade(self, symbol: str, signal: str, confidence: float, 
                       current_price: float, metadata: dict):
        """Execute a trade based on regime strategy signal"""
        try:
            # Check if we can open a position
            can_open, reason = self.risk_manager.can_open_position(
                self.risk_manager.current_capital * Config.RISK_PER_TRADE
            )
            
            if not can_open:
                logger.warning(f"{symbol}: Cannot open position - {reason}")
                return
            
            # Check existing positions
            positions = self.mt5.get_positions(symbol)
            if positions and len(positions) > 0:
                logger.info(f"{symbol}: Position already open, skipping")
                return
            
            # Get stop loss and take profit from metadata
            stop_loss = metadata.get('stop_loss')
            take_profit = metadata.get('take_profit')
            
            if not stop_loss or not take_profit:
                logger.warning(f"{symbol}: Missing stop loss or take profit")
                return
            
            # Calculate position size using the multiplier from strategy
            size_multiplier = metadata.get('position_size_multiplier', 1.0)
            base_risk = self.risk_manager.current_capital * self.strategy.base_risk_percent
            adjusted_risk = base_risk * size_multiplier
            
            # Calculate lot size
            risk_per_unit = abs(current_price - stop_loss)
            if risk_per_unit > 0:
                position_size = adjusted_risk / risk_per_unit
            else:
                position_size = 0.01  # Minimum size
            
            # Clamp to valid range
            position_size = max(0.01, min(position_size, 1.0))
            
            logger.info(
                f"📊 {symbol} Trade Setup:\n"
                f"   Signal: {signal} | Confidence: {confidence:.1%}\n"
                f"   Entry: {current_price:.5f}\n"
                f"   Stop Loss: {stop_loss:.5f}\n"
                f"   Take Profit: {take_profit:.5f}\n"
                f"   Size Multiplier: {size_multiplier:.2f}\n"
                f"   Position Size: {position_size:.2f} lots\n"
                f"   Regime: {metadata.get('regime')}\n"
                f"   Max Hold: {metadata.get('max_hold_bars', 'N/A')} bars"
            )
            
            # Execute the trade
            order_type = 'BUY' if signal == 'BUY' else 'SELL'
            
            result = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                price=current_price,
                sl=stop_loss,
                tp=take_profit,
                comment=f"Regime_{metadata.get('regime', 'unknown')[:10]}"
            )
            
            if result and result.get('success'):
                ticket = result.get('order')
                logger.success(
                    f"✅ {symbol}: {order_type} order placed | Ticket: {ticket} | "
                    f"Size: {position_size:.2f}"
                )
                
                # Track position
                self.open_positions[symbol] = {
                    'ticket': ticket,
                    'signal': signal,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'regime': metadata.get('regime'),
                    'entry_time': datetime.now(),
                    'max_hold_bars': metadata.get('max_hold_bars', 50),
                    'exit_conditions': metadata.get('exit_conditions', [])
                }
            else:
                logger.error(f"❌ {symbol}: Order failed - {result}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def check_open_positions(self):
        """Check and manage open positions"""
        try:
            for symbol, position_info in list(self.open_positions.items()):
                positions = self.mt5.get_positions(symbol)
                
                if not positions or len(positions) == 0:
                    # Position was closed
                    logger.info(f"{symbol}: Position closed")
                    
                    # Try to determine if win or loss
                    # This is a simplified check - you might want to enhance this
                    del self.open_positions[symbol]
                    continue
                
                # Position still open - check exit conditions
                # Time-based exit check
                entry_time = position_info.get('entry_time')
                if entry_time:
                    # Simplified time check - could enhance with bar counting
                    hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                    max_hours = position_info.get('max_hold_bars', 50) / 10  # Rough conversion
                    
                    if hours_held > max_hours:
                        logger.warning(f"{symbol}: Time limit reached, closing position")
                        # Close position logic here
                        
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def record_trade_result(self, symbol: str, pnl: float):
        """Record trade result for strategy state management"""
        is_win = pnl > 0
        self.strategy.record_trade_result(pnl, is_win)
        
        status = "WIN" if is_win else "LOSS"
        logger.info(f"{symbol}: Trade closed - {status} (${pnl:.2f})")
        
        # Check if kill switch was triggered
        summary = self.strategy.get_regime_summary()
        if summary['kill_switch_active']:
            logger.warning(f"⚠️ Kill switch activated: {summary['kill_switch_reason']}")
    
    def run(self, symbols: list = None):
        """Main trading loop"""
        if symbols is None:
            symbols = Config.TRADING_SYMBOLS
        
        logger.info(f"Starting bot with symbols: {symbols}")
        
        # Initialize
        self.initialize()
        
        # Start data collection
        self.start_data_collection(symbols)
        
        # Wait for initial data
        logger.info("Waiting for initial data collection (30 seconds)...")
        time.sleep(30)
        
        self.is_running = True
        
        logger.info("=" * 60)
        logger.info("Trading loop started - Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        try:
            loop_count = 0
            while self.is_running:
                loop_count += 1
                
                # Log regime summary every 10 loops
                if loop_count % 10 == 0:
                    summary = self.strategy.get_regime_summary()
                    logger.info(
                        f"📈 Regime Summary | Current: {summary['current_regime']} | "
                        f"Age: {summary['regime_age_bars']} bars | "
                        f"Daily trades: {summary['daily_trades']}/{self.strategy.max_daily_trades}"
                    )
                    
                    if summary['kill_switch_active']:
                        logger.warning(f"⚠️ Kill switch active: {summary['kill_switch_reason']}")
                
                # Analyze each symbol
                for symbol in symbols:
                    self.analyze_and_trade(symbol)
                
                # Check open positions
                self.check_open_positions()
                
                # Sleep between iterations
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup on exit"""
        self.is_running = False
        
        # Stop data handlers
        for symbol, handler in self.market_data.items():
            try:
                handler.stop()
            except:
                pass
        
        # Disconnect MT5
        if self.mt5:
            self.mt5.disconnect()
        
        logger.info("Bot stopped")


def main():
    """Main entry point"""
    bot = DerivBotRegime()
    
    # You can specify symbols here or use config
    # Default to volatility indices for testing
    symbols = ['Volatility 10 Index', 'Volatility 25 Index']
    
    # Check if symbols provided as args
    if len(sys.argv) > 1:
        symbols = sys.argv[1].split(',')
    
    bot.run(symbols)


if __name__ == "__main__":
    main()
