"""
Regime-Aware Trading Bot using Deriv WebSocket API
===================================================
Alternative to MT5 version - works on Linux/Render

This file uses Deriv's WebSocket API instead of MetaTrader5.
Your original regime_trading_bot.py (MT5 version) is preserved.

To revert: Just use regime_trading_bot.py instead of this file.
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from loguru import logger

# Import Deriv API
from api.deriv_api import DerivAPI

# Import strategy
from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime, TradabilityState


@dataclass
class Position:
    """Represents an open position/contract"""
    contract_id: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    stake: float
    entry_price: float
    entry_time: datetime
    regime: str
    max_hold_bars: int
    bars_held: int = 0


class RegimeTradingBotDeriv:
    """
    Automated trading bot using Deriv WebSocket API
    
    Works on Linux (Render, Railway, etc.)
    """
    
    # Symbol mapping: Deriv API symbol names
    SYMBOL_MAP = {
        'Volatility 10 Index': 'R_10',
        'Volatility 25 Index': 'R_25',
        'Volatility 50 Index': 'R_50',
        'Volatility 75 Index': 'R_75',
        'Volatility 100 Index': 'R_100',
        'Boom 1000 Index': 'BOOM1000',
        'Crash 1000 Index': 'CRASH1000',
    }
    
    def __init__(self, symbols: List[str], risk_per_trade: float = None):
        """
        Initialize the trading bot
        
        Args:
            symbols: List of symbols to trade (MT5-style names)
            risk_per_trade: Risk per trade as decimal (0.01 = 1%)
        """
        self.symbols = symbols
        
        # Load from environment or use defaults
        if risk_per_trade is None:
            risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.01'))
        self.risk_per_trade = risk_per_trade
        
        # One strategy instance per symbol
        self.strategies: Dict[str, RegimeAwareStrategy] = {}
        for symbol in symbols:
            self.strategies[symbol] = RegimeAwareStrategy()
        
        # Open positions tracking
        self.positions: Dict[str, Position] = {}
        
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
        
        # Deriv API
        self.api: Optional[DerivAPI] = None
        
        # Price data cache
        self.price_data: Dict[str, pd.DataFrame] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging"""
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            "logs/deriv_bot_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="1 day"
        )
    
    def connect(self) -> bool:
        """Connect to Deriv API"""
        app_id = os.getenv('DERIV_APP_ID')
        api_token = os.getenv('DERIV_API_TOKEN')
        
        if not app_id or not api_token:
            logger.error("DERIV_APP_ID and DERIV_API_TOKEN must be set")
            return False
        
        try:
            self.api = DerivAPI(app_id, api_token)
            self.api.connect()
            
            # Get balance
            balance = self.api.get_account_balance()
            self.initial_balance = balance
            
            logger.info(f"Connected to Deriv API")
            logger.info(f"Balance: ${balance:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Deriv API"""
        if self.api:
            self.api.close()
        logger.info("Disconnected from Deriv API")
    
    def get_symbol_code(self, symbol: str) -> str:
        """Convert MT5-style symbol name to Deriv API code"""
        return self.SYMBOL_MAP.get(symbol, symbol)
    
    def get_ohlc_data(self, symbol: str, bars: int = 200) -> Optional[pd.DataFrame]:
        """Get OHLC data from Deriv API"""
        try:
            deriv_symbol = self.get_symbol_code(symbol)
            
            # Get candles (1 minute)
            response = self.api.send_request({
                "ticks_history": deriv_symbol,
                "count": bars,
                "end": "latest",
                "style": "candles",
                "granularity": 60  # 1 minute candles
            })
            
            if 'candles' not in response:
                return None
            
            candles = response['candles']
            
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close'
            }, inplace=True)
            
            # Add volume (not available in Deriv, use placeholder)
            df['volume'] = 1000
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error getting OHLC for {symbol}: {e}")
            return None
    
    def calculate_stake(self, multiplier: float = 1.0) -> float:
        """
        Calculate stake based on risk
        
        For Deriv contracts, we stake a fixed amount.
        Position sizing is simpler than MT5.
        """
        balance = self.api.get_account_balance()
        stake = balance * self.risk_per_trade * multiplier
        
        # Minimum stake
        stake = max(1.0, stake)
        # Maximum stake per trade
        stake = min(stake, balance * 0.05)  # Max 5% of balance
        
        return round(stake, 2)
    
    def open_trade(self, symbol: str, direction: str, stake: float,
                   regime: str, max_hold_bars: int, duration: int = 5) -> Optional[int]:
        """
        Open a trade using Deriv API
        
        For synthetic indices, we use CALL/PUT contracts
        """
        try:
            deriv_symbol = self.get_symbol_code(symbol)
            
            # Map direction to contract type
            contract_type = 'CALL' if direction == 'BUY' else 'PUT'
            
            # Get current price for reference
            tick_response = self.api.send_request({
                "ticks": deriv_symbol
            })
            current_price = tick_response.get('tick', {}).get('quote', 0)
            
            # Buy contract
            # Using duration in minutes for more control
            result = self.api.buy_contract(
                contract_type=contract_type,
                symbol=deriv_symbol,
                amount=stake,
                duration=duration,
                duration_unit='m'  # minutes
            )
            
            if 'error' in result:
                logger.error(f"Trade failed: {result['error']}")
                return None
            
            contract_id = result.get('contract_id')
            
            # Track position
            self.positions[symbol] = Position(
                contract_id=contract_id,
                symbol=symbol,
                direction=direction,
                stake=stake,
                entry_price=current_price,
                entry_time=datetime.now(),
                regime=regime,
                max_hold_bars=max_hold_bars
            )
            
            self.daily_trades += 1
            
            logger.info(f"✅ OPENED: {contract_type} {symbol} | Stake: ${stake:.2f}")
            logger.info(f"   Contract ID: {contract_id}")
            
            return contract_id
            
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
            return None
    
    def check_open_positions(self):
        """Check and manage open positions"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            position.bars_held += 1
            
            # For Deriv contracts, they expire automatically
            # Just track time-based exits
            if position.bars_held >= position.max_hold_bars:
                logger.info(f"Max hold reached for {symbol}")
                # Contract will expire on its own
                del self.positions[symbol]
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits are exceeded"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            
            for strategy in self.strategies.values():
                strategy.reset_kill_switch()
        
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Check daily loss
        balance = self.api.get_account_balance()
        daily_return = (balance - self.initial_balance) / self.initial_balance
        if daily_return < -self.max_daily_loss:
            logger.warning(f"Daily loss limit: {daily_return:.2%}")
            return False
        
        return True
    
    def analyze_and_trade(self, symbol: str):
        """Analyze a symbol and trade if signal found"""
        if symbol in self.positions:
            return
        
        strategy = self.strategies[symbol]
        
        df = self.get_ohlc_data(symbol)
        if df is None or len(df) < 150:
            return
        
        current_price = df['close'].iloc[-1]
        
        # Analyze
        signal, confidence, metadata = strategy.analyze(df, current_price)
        
        if signal != 'NEUTRAL' and confidence >= (Config.MIN_CONFIDENCE_SCORE * 100):
            logger.info(f"{symbol}: {signal} signal ({confidence:.0f}%) - {metadata.get('regime', 'unknown')}")
            logger.info(f"   Reason: {metadata.get('reason', 'N/A')}")
            
            multiplier = metadata.get('position_size_multiplier', 1.0)
            stake = self.calculate_stake(multiplier)
            max_hold = metadata.get('max_hold_bars', 50)
            
            # Open trade
            contract_id = self.open_trade(
                symbol=symbol,
                direction=signal,
                stake=stake,
                regime=metadata.get('regime', 'unknown'),
                max_hold_bars=max_hold,
                duration=5  # 5 minute contracts
            )
            
            if contract_id:
                logger.info(f"   Trade opened successfully!")
    
    def print_status(self):
        """Print current status"""
        balance = self.api.get_account_balance()
        pnl = balance - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        print("\n" + "=" * 70)
        print(f"💰 Balance: ${balance:.2f} | 📈 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"📊 Daily trades: {self.daily_trades}/{self.max_daily_trades}")
        print(f"📈 Open positions: {len(self.positions)}")
        print("-" * 70)
        
        # Symbol status
        print(f"{'Symbol':<25} {'Regime':<18} {'Age':>5} {'Vol%':>6}")
        print("-" * 70)
        
        for symbol in self.symbols:
            strategy = self.strategies[symbol]
            df = self.get_ohlc_data(symbol, bars=150)
            
            if df is not None:
                try:
                    from utils.regime_detector import calculate_volatility_metrics
                    vol_metrics = calculate_volatility_metrics(df)
                    vol_pct = f"{vol_metrics['volatility_percentile']:.0f}%" if vol_metrics['is_valid'] else "N/A"
                except:
                    vol_pct = "N/A"
                
                regime = strategy.current_regime.value if strategy.current_regime else "unknown"
                age = strategy.regime_bar_count
                
                print(f"{symbol:<25} {regime:<18} {age:>5} {vol_pct:>6}")
        
        print("=" * 70)
        
        if self.positions:
            print("\n📋 OPEN POSITIONS:")
            for symbol, pos in self.positions.items():
                print(f"   {'🟢' if pos.direction == 'BUY' else '🔴'} {pos.direction} {symbol}: "
                      f"Bars: {pos.bars_held}/{pos.max_hold_bars}")
    
    def run(self, interval: int = 60):
        """
        Main bot loop
        
        Args:
            interval: Seconds between analysis cycles
        """
        if not self.connect():
            logger.error("Failed to connect. Exiting.")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Bot started - monitoring {len(self.symbols)} symbols")
        logger.info(f"Risk per trade: {self.risk_per_trade:.1%}")
        logger.info(f"Max daily trades: {self.max_daily_trades}")
        
        print("\n" + "=" * 70)
        print("🤖 REGIME TRADING BOT (DERIV API) STARTED")
        print("=" * 70)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Risk per trade: {self.risk_per_trade:.1%}")
        print(f"Analysis interval: {interval} seconds")
        print("\nPress Ctrl+C to stop")
        print("=" * 70)
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                
                # Check daily limits
                if not self.check_daily_limits():
                    time.sleep(interval)
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
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.is_running = False
            self.disconnect()


def main():
    """Main entry point"""
    # Symbols to trade
    symbols = [
        'Volatility 50 Index',
        'Volatility 75 Index',
        'Volatility 100 Index',
    ]
    
    # Create and run bot
    bot = RegimeTradingBotDeriv(
        symbols=symbols,
        risk_per_trade=None  # Will read from env or use default 1%
    )
    
    bot.run(interval=60)


if __name__ == "__main__":
    main()
