"""
MT5 Trading Bot with Price Action Strategy
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
from price_action_strategy import PriceActionStrategy

# Configure logger
logger.remove()
logger.add(sys.stderr, level=Config.LOG_LEVEL)
logger.add(
    Config.LOG_DIR / f"trading_mt5_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="1 day",
    retention="30 days",
    level=Config.LOG_LEVEL
)


class DerivBotMT5:
    """MT5-based trading bot with price action strategy"""
    
    def __init__(self):
        self.mt5: MT5API = None
        self.market_data: dict = {}
        self.strategy = None
        self.risk_manager: RiskManager = None
        self.is_running = False
        
        # Trade tracking
        self.open_positions = {}
        self.pending_signals = {}  # Track signals waiting for confirmation
        
        logger.info("=" * 60)
        logger.info("DerivBot MT5 - Price Action Trading")
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
            
            # Initialize risk manager with configured capital (not MT5 balance)
            trading_capital = Config.INITIAL_CAPITAL if Config.INITIAL_CAPITAL > 0 else balance
            self.risk_manager = RiskManager(trading_capital)
            logger.info(f"Risk manager initialized with ${trading_capital:.2f} capital")
            
            if trading_capital != balance:
                logger.warning(f"⚠️  Trading with ${trading_capital:.2f} (configured) instead of ${balance:.2f} (MT5 balance)")
            
            # Initialize price action strategy
            logger.info("Initializing price action strategy")
            self.strategy = PriceActionStrategy()
            
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
    
    def analyze_and_trade(self, symbol: str):
        """Analyze symbol and execute trades"""
        try:
            handler = self.market_data.get(symbol)
            if not handler:
                return
            
            # Update with latest data
            handler.update()
            
            # Get data
            df = handler.get_dataframe()
            
            if len(df) < 200:  # Need enough data for analysis
                return
            
            # Add technical indicators
            df = TechnicalIndicators.calculate_all(df)
            
            # Get strategy signal
            result = self.strategy.analyze(df, symbol)
            signal = result['signal']
            confidence = result['confidence']
            reason = result['reason']
            
            current_price = df.iloc[-1]['close']
            
            logger.info(f"{symbol} | Signal: {signal} | Confidence: {confidence:.2f}% | Price: {current_price:.5f}")
            
            # Execute trade if signal is strong enough
            if signal in ['BUY', 'SELL'] and confidence >= Config.MIN_CONFIDENCE_SCORE:
                # Check entry timing filters
                if self._check_entry_timing(symbol, signal, df, current_price, result):
                    self._execute_trade(symbol, signal, confidence, current_price, df)
                else:
                    logger.debug(f"{symbol}: Signal found but waiting for better entry timing")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    def _check_entry_timing(self, symbol: str, signal: str, df, current_price: float, result: dict) -> bool:
        """Check if entry timing is good - near structure + confirmation candle"""
        try:
            # FILTER 1: Price must be near structure level
            analysis = self.strategy.symbol_analysis.get(symbol, {})
            supports = analysis.get('support_levels', [])
            resistances = analysis.get('resistance_levels', [])
            
            atr = df.iloc[-1].get('atr', current_price * 0.01)
            structure_tolerance = atr * 1.5  # Within 1.5 ATR of structure
            
            near_structure = False
            
            if signal == 'BUY':
                # Check if near support
                for support in supports:
                    if abs(current_price - support) <= structure_tolerance:
                        near_structure = True
                        logger.debug(f"{symbol}: Near support {support:.5f} (tolerance {structure_tolerance:.5f})")
                        break
            else:  # SELL
                # Check if near resistance
                for resistance in resistances:
                    if abs(current_price - resistance) <= structure_tolerance:
                        near_structure = True
                        logger.debug(f"{symbol}: Near resistance {resistance:.5f} (tolerance {structure_tolerance:.5f})")
                        break
            
            if not near_structure:
                logger.debug(f"{symbol}: Price not near structure - waiting for pullback")
                return False
            
            # FILTER 2: Wait for confirmation candle
            # Check if we have a pending signal waiting for confirmation
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            signal_key = f"{symbol}_{signal}"
            
            # Check if this is a new signal
            if signal_key not in self.pending_signals:
                # New signal - wait for confirmation candle
                self.pending_signals[signal_key] = {
                    'time': datetime.now(),
                    'entry_price': current_price,
                    'signal': signal
                }
                logger.info(f"⏳ {symbol}: New {signal} signal - waiting for confirmation candle")
                return False
            
            # Signal exists - check if we have confirmation candle
            if signal == 'BUY':
                # Confirmation: current candle closed bullish
                confirmation = current['close'] > current['open']
            else:  # SELL
                # Confirmation: current candle closed bearish
                confirmation = current['close'] < current['open']
            
            if confirmation:
                logger.success(f"✅ {symbol}: Confirmation candle received - entering {signal}")
                # Clear pending signal
                del self.pending_signals[signal_key]
                return True
            else:
                # Check if signal is still valid (within 2 minutes)
                pending = self.pending_signals[signal_key]
                elapsed = (datetime.now() - pending['time']).total_seconds()
                if elapsed > 120:  # 2 minutes timeout
                    logger.warning(f"⏰ {symbol}: Signal expired without confirmation")
                    del self.pending_signals[signal_key]
                    return False
                
                logger.debug(f"{symbol}: Still waiting for confirmation candle...")
                return False
                
        except Exception as e:
            logger.error(f"Error checking entry timing: {e}")
            return True  # Default to allowing trade if check fails
    
    def _execute_trade(self, symbol: str, signal: str, confidence: float, 
                      entry_price: float, df):
        """Execute a trade on MT5"""
        try:
            # Get structure-based stops from strategy
            if hasattr(self.strategy, 'get_structure_stop_loss'):
                stop_loss_price = self.strategy.get_structure_stop_loss(entry_price, signal, symbol)
                take_profit_price = self.strategy.get_structure_take_profit(entry_price, signal, symbol)
                
                # Reject if no clear target (TP is None)
                if take_profit_price is None:
                    logger.warning(f"❌ Skipping trade: No clear resistance/support target")
                    return
                
                logger.info(f"Using structure-based stops: SL={stop_loss_price:.5f}, TP={take_profit_price:.5f}")
            else:
                # Fallback to risk manager
                atr = df.iloc[-1].get('atr', None)
                stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, signal, atr)
                take_profit_price = self.risk_manager.calculate_take_profit(entry_price, stop_loss_price, signal)
            
            # Calculate R:R and validate minimum requirement
            risk_pips = abs(entry_price - stop_loss_price)
            reward_pips = abs(take_profit_price - entry_price)
            risk_reward = reward_pips / risk_pips if risk_pips > 0 else 0
            
            # Minimum R:R filter (3:1 for quality over quantity)
            if risk_reward < 3.0:
                logger.warning(f"❌ Skipping trade: Poor R:R ratio {risk_reward:.2f}:1 (need 3:1 minimum)")
                logger.warning(f"   Risk: {risk_pips:.5f} | Reward: {reward_pips:.5f}")
                return
            
            logger.info(f"✅ R:R ratio: {risk_reward:.2f}:1")
            
            # Calculate position size (volume in lots)
            risk_amount = self.risk_manager.current_capital * Config.RISK_PER_TRADE
            
            # Get symbol info for volume calculation
            symbol_info = self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol}")
                return
            
            # Calculate volume (lots)
            # For synthetic indices, 1 lot = 1 unit
            volume = risk_amount / risk_pips
            
            # Round to symbol's volume step (e.g., 0.5 for 1HZ10V, 0.01 for others)
            volume_step = symbol_info.get('volume_step', 0.01)
            volume = max(symbol_info['volume_min'], round(volume / volume_step) * volume_step)
            volume = min(volume, symbol_info['volume_max'])
            
            logger.debug(f"Symbol {symbol}: min_vol={symbol_info['volume_min']}, step={volume_step}, calculated={volume}")
            
            # Calculate actual risk with minimum volume
            actual_risk = volume * risk_pips
            actual_risk_percent = (actual_risk / self.risk_manager.current_capital) * 100
            
            # Risk cap: Skip if actual risk exceeds 2x intended risk
            max_acceptable_risk = risk_amount * 2.0
            if actual_risk > max_acceptable_risk:
                logger.warning(f"❌ Skipping trade: Actual risk ${actual_risk:.2f} ({actual_risk_percent:.1f}%) exceeds max ${max_acceptable_risk:.2f}")
                logger.warning(f"   Minimum volume {volume} lots too large for capital ${self.risk_manager.current_capital:.2f}")
                logger.warning(f"   💡 Consider: Increase capital or reduce RISK_PER_TRADE")
                return
            
            logger.info(f"💰 Risk: ${actual_risk:.2f} ({actual_risk_percent:.1f}% of capital)")
            
            # Check if can open position
            can_open, reason = self.risk_manager.can_open_position(actual_risk)
            
            if not can_open:
                logger.warning(f"Cannot open position: {reason}")
                return
            
            logger.info(f"Opening {signal} position on {symbol}")
            logger.info(f"Entry: {entry_price:.5f} | SL: {stop_loss_price:.5f} | TP: {take_profit_price:.5f}")
            logger.info(f"Volume: {volume} lots | Confidence: {confidence:.2f}%")
            
            # Execute trade
            if signal == 'BUY':
                result = self.mt5.buy(
                    symbol=symbol,
                    volume=volume,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    comment=f"PA {confidence:.0f}%"
                )
            else:
                result = self.mt5.sell(
                    symbol=symbol,
                    volume=volume,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    comment=f"PA {confidence:.0f}%"
                )
            
            if result:
                ticket = result['ticket']
                self.open_positions[ticket] = {
                    'symbol': symbol,
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'volume': volume,
                    'time': datetime.now()
                }
                
                self.risk_manager.open_positions += 1
                logger.success(f"✅ Trade executed | Ticket: {ticket}")
            else:
                logger.error(f"❌ Trade execution failed")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            # Get current positions from MT5
            mt5_positions = self.mt5.get_open_positions()
            
            # Update risk manager position count
            self.risk_manager.open_positions = len(mt5_positions)
            
            # Check for closed positions and record results
            open_tickets = {pos['ticket'] for pos in mt5_positions}
            closed_tickets = set(self.open_positions.keys()) - open_tickets
            
            for ticket in closed_tickets:
                pos_info = self.open_positions[ticket]
                
                # Try to get the closed position result (profit/loss)
                # This is simplified - in production you'd track this more carefully
                logger.info(f"Position {ticket} closed for {pos_info['symbol']}")
                
                # Remove from tracking
                del self.open_positions[ticket]
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def run(self):
        """Main trading loop"""
        try:
            logger.info("Starting trading bot...")
            
            # Start data collection
            symbols = [s.strip() for s in Config.TRADING_SYMBOLS]
            self.start_data_collection(symbols)
            
            # Wait for initial data
            logger.info("Collecting initial data (10 seconds)...")
            time.sleep(10)
            
            logger.success("Bot is now running!")
            logger.info("Monitoring markets and looking for opportunities...")
            
            self.is_running = True
            
            while self.is_running:
                logger.debug(f"Loop iteration - is_running={self.is_running}")
                # Analyze each symbol
                for symbol in symbols:
                    try:
                        self.analyze_and_trade(symbol)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Monitor open positions
                try:
                    self.monitor_positions()
                except Exception as e:
                    logger.error(f"Error monitoring positions: {e}")
                
                # Wait before next analysis cycle
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Shutting down bot...")
            self.is_running = False
            
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            raise
            
        finally:
            # Cleanup
            if self.mt5:
                self.mt5.close()
            logger.info("Bot stopped")


if __name__ == "__main__":
    bot = DerivBotMT5()
    
    try:
        bot.initialize()
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
