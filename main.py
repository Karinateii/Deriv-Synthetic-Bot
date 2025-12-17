"""
Main Trading Bot Controller
"""
import sys
import time
from datetime import datetime
from loguru import logger
from pathlib import Path

from config import Config
from api.deriv_api import DerivAPI
from api.market_data import MarketDataHandler
from strategies.ml_strategy import MLStrategy
from strategies.multi_indicator import MultiIndicatorStrategy
from risk.risk_manager import RiskManager
from utils.indicators import TechnicalIndicators

# Configure logger
logger.remove()
logger.add(sys.stderr, level=Config.LOG_LEVEL)
logger.add(
    Config.LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log",
    rotation="1 day",
    retention="30 days",
    level=Config.LOG_LEVEL
)


class DerivBot:
    """Main trading bot controller"""
    
    def __init__(self):
        self.api: DerivAPI = None
        self.market_data: dict[str, MarketDataHandler] = {}
        self.strategy: MLStrategy = None
        self.risk_manager: RiskManager = None
        self.is_running = False
        
        # Trade tracking
        self.open_positions = {}
        self.trade_history = []
        
        logger.info("=" * 60)
        logger.info("DerivBot - AI-Powered Trading Bot")
        logger.info("=" * 60)
        
    def initialize(self):
        """Initialize bot components"""
        try:
            # Validate configuration
            Config.validate()
            
            # Connect to Deriv API
            logger.info("Connecting to Deriv API...")
            self.api = DerivAPI(Config.DERIV_APP_ID, Config.DERIV_API_TOKEN)
            self.api.connect()
            
            # Get account balance
            balance = self.api.get_account_balance()
            logger.success(f"Account balance: ${balance:.2f}")
            
            # Initialize risk manager
            self.risk_manager = RiskManager(balance)
            logger.info("Risk manager initialized")
            
            # Initialize strategy
            logger.info(f"Initializing strategy: {Config.PRIMARY_STRATEGY}")
            
            if Config.PRIMARY_STRATEGY == 'ml_ensemble':
                self.strategy = MLStrategy()
            else:
                self.strategy = MultiIndicatorStrategy()
            
            logger.success("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def start_data_collection(self, symbols: list[str]):
        """Start collecting market data for symbols"""
        for symbol in symbols:
            try:
                logger.info(f"Starting data collection for {symbol}")
                handler = MarketDataHandler(self.api, symbol)
                handler.start()
                
                # Add tick callback
                handler.add_tick_callback(lambda price, timestamp, sym=symbol: 
                                         self._on_tick(sym, price, timestamp))
                
                self.market_data[symbol] = handler
                logger.success(f"Data collection started for {symbol}")
                
            except Exception as e:
                logger.error(f"Error starting data collection for {symbol}: {e}")
    
    def train_models(self):
        """Train ML models on historical data"""
        if not Config.ENABLE_AI_PREDICTION:
            logger.info("AI prediction disabled, skipping training")
            return
        
        try:
            logger.info("Training ML models on historical data...")
            
            # Use first symbol for training
            if self.market_data:
                symbol = list(self.market_data.keys())[0]
                handler = self.market_data[symbol]
                
                # Get historical data
                df = handler.get_ohlc('1min', periods=2000)
                
                if len(df) >= Config.MIN_TRAINING_SAMPLES:
                    # Calculate indicators
                    df = TechnicalIndicators.calculate_all(df)
                    
                    # Train strategy
                    if isinstance(self.strategy, MLStrategy):
                        self.strategy.train(df)
                        logger.success("ML models trained successfully")
                else:
                    logger.warning("Insufficient data for training")
                    
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _on_tick(self, symbol: str, price: float, timestamp: int):
        """Handle new tick data"""
        # This is called for every tick - keep it lightweight
        pass
    
    def analyze_and_trade(self, symbol: str):
        """Analyze market and execute trades"""
        try:
            handler = self.market_data.get(symbol)
            if not handler:
                return
            
            # Check if enough data
            if not handler.is_data_ready(min_ticks=200):
                logger.debug(f"{symbol}: Waiting for more data...")
                return
            
            # Get market data
            df = handler.get_ohlc('1min', periods=200)
            current_price = handler.get_current_price()
            
            if df.empty or current_price == 0:
                return
            
            # Calculate indicators
            df = TechnicalIndicators.calculate_all(df)
            
            # Get signal from strategy
            signal, confidence, metadata = self.strategy.get_signal(df, current_price)
            
            logger.info(f"{symbol} | Signal: {signal} | Confidence: {confidence:.2%} | Price: {current_price:.5f}")
            
            # Execute trade if signal is strong enough
            if signal in ['BUY', 'SELL'] and confidence >= Config.MIN_CONFIDENCE_SCORE:
                self._execute_trade(symbol, signal, confidence, current_price, df, metadata)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    def _execute_trade(self, symbol: str, signal: str, confidence: float, 
                      entry_price: float, df, metadata: dict):
        """Execute a trade"""
        try:
            # Get structure-based stops from strategy if available
            atr = df.iloc[-1].get('atr', None)
            
            if hasattr(self.strategy, 'get_structure_stop_loss'):
                # Use structure-based levels from price action analysis
                stop_loss_price = self.strategy.get_structure_stop_loss(entry_price, signal)
                take_profit_price = self.strategy.get_structure_take_profit(entry_price, signal)
                logger.info(f"Using structure-based stops: SL={stop_loss_price:.5f}, TP={take_profit_price:.5f}")
            else:
                # Fallback to risk manager calculation
                stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, signal, atr)
                take_profit_price = self.risk_manager.calculate_take_profit(entry_price, stop_loss_price, signal)
            
            # Calculate position size
            stake_amount = self.risk_manager.calculate_position_size(
                entry_price, stop_loss_price, method='fixed_percent'
            )
            
            # Check if can open position
            can_open, reason = self.risk_manager.can_open_position(stake_amount)
            
            if not can_open:
                logger.warning(f"Cannot open position: {reason}")
                return
            
            # Determine contract type
            if signal == 'BUY':
                contract_type = 'CALL'
            else:
                contract_type = 'PUT'
            
            logger.info(f"Opening {contract_type} position on {symbol}")
            logger.info(f"Entry: {entry_price:.5f} | SL: {stop_loss_price:.5f} | TP: {take_profit_price:.5f}")
            logger.info(f"Stake: ${stake_amount:.2f} | Confidence: {confidence:.2%}")
            
            # Execute trade
            result = self.api.buy_contract(
                contract_type=contract_type,
                symbol=symbol,
                amount=stake_amount,
                duration=10,
                duration_unit='ticks'
            )
            
            if 'contract_id' in result:
                contract_id = result['contract_id']
                
                # Track position
                self.open_positions[contract_id] = {
                    'symbol': symbol,
                    'type': contract_type,
                    'entry_price': entry_price,
                    'stake': stake_amount,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'confidence': confidence,
                    'entry_time': datetime.now(),
                    'metadata': metadata
                }
                
                self.risk_manager.open_positions += 1
                
                logger.success(f"Position opened: Contract ID {contract_id}")
            else:
                logger.error(f"Failed to open position: {result}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            # Get open positions from API
            positions = self.api.get_open_positions()
            
            for position in positions:
                contract_id = position.get('contract_id')
                current_price = float(position.get('current_spot', 0))
                pnl = float(position.get('profit', 0))
                
                if contract_id in self.open_positions:
                    pos_info = self.open_positions[contract_id]
                    
                    # Check stop loss / take profit
                    should_close = False
                    close_reason = ""
                    
                    if pos_info['type'] == 'CALL':
                        if current_price <= pos_info['stop_loss']:
                            should_close = True
                            close_reason = "Stop loss hit"
                        elif current_price >= pos_info['take_profit']:
                            should_close = True
                            close_reason = "Take profit hit"
                    else:  # PUT
                        if current_price >= pos_info['stop_loss']:
                            should_close = True
                            close_reason = "Stop loss hit"
                        elif current_price <= pos_info['take_profit']:
                            should_close = True
                            close_reason = "Take profit hit"
                    
                    if should_close:
                        self._close_position(contract_id, current_price, close_reason)
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def _close_position(self, contract_id: int, current_price: float, reason: str):
        """Close a position"""
        try:
            pos_info = self.open_positions.get(contract_id)
            if not pos_info:
                return
            
            logger.info(f"Closing position {contract_id}: {reason}")
            
            # Sell contract
            result = self.api.sell_contract(contract_id, current_price)
            
            if 'sold_for' in result:
                sold_price = float(result['sold_for'])
                pnl = sold_price - pos_info['stake']
                is_win = pnl > 0
                
                # Record trade
                self.risk_manager.record_trade(pnl, is_win)
                self.risk_manager.open_positions -= 1
                
                # Log result
                logger.info(f"Position closed: PnL ${pnl:.2f} ({'+' if pnl > 0 else ''}{(pnl/pos_info['stake']*100):.2f}%)")
                
                # Record in history
                trade_record = {
                    **pos_info,
                    'exit_price': current_price,
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'pnl_percent': (pnl / pos_info['stake']) * 100,
                    'close_reason': reason
                }
                self.trade_history.append(trade_record)
                
                # Remove from open positions
                del self.open_positions[contract_id]
                
            else:
                logger.error(f"Failed to close position: {result}")
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def run(self):
        """Main trading loop"""
        try:
            logger.info("Starting trading bot...")
            self.is_running = True
            
            # Start data collection
            self.start_data_collection(Config.TRADING_SYMBOLS)
            
            # Wait for initial data
            logger.info("Collecting initial data (30 seconds)...")
            time.sleep(30)
            
            # Train models
            self.train_models()
            
            logger.success("Bot is now running!")
            logger.info("Monitoring markets and looking for opportunities...")
            
            iteration = 0
            
            while self.is_running:
                iteration += 1
                
                # Analyze each symbol
                for symbol in Config.TRADING_SYMBOLS:
                    self.analyze_and_trade(symbol)
                
                # Monitor open positions
                self.monitor_positions()
                
                # Print stats every 100 iterations
                if iteration % 100 == 0:
                    self._print_stats()
                
                # Sleep between iterations
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()
    
    def _print_stats(self):
        """Print current statistics"""
        stats = self.risk_manager.get_stats()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE STATS")
        logger.info(f"Capital: ${stats['current_capital']:.2f} | "
                   f"PnL: ${stats['total_pnl']:.2f} ({stats['total_pnl_percent']:+.2f}%)")
        logger.info(f"Trades: {stats['total_trades']} | "
                   f"Win Rate: {stats['win_rate']:.1f}% | "
                   f"Drawdown: {stats['current_drawdown']:.2f}%")
        logger.info(f"Open Positions: {stats['open_positions']}")
        logger.info("=" * 60)
    
    def stop(self):
        """Stop the bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Close all open positions
        if self.open_positions:
            logger.info("Closing all open positions...")
            for contract_id in list(self.open_positions.keys()):
                current_price = 0  # Would need to get from API
                self._close_position(contract_id, current_price, "Bot shutdown")
        
        # Disconnect API
        if self.api:
            self.api.close()
        
        # Print final stats
        self._print_stats()
        
        logger.success("Bot stopped successfully")


def main():
    """Main entry point"""
    try:
        bot = DerivBot()
        bot.initialize()
        bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
