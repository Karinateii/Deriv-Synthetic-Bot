"""
Simple Backtesting Script
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import sys

from config import Config
from utils.indicators import TechnicalIndicators
from strategies.multi_indicator import MultiIndicatorStrategy
from strategies.ml_strategy import MLStrategy
from strategies.high_accuracy import HighAccuracyStrategy

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


class SimpleBacktest:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        
    def generate_synthetic_data(self, periods: int = 2000) -> pd.DataFrame:
        """
        Generate synthetic price data for testing
        
        In production, you would load real historical data from Deriv
        """
        logger.info("Generating synthetic market data...")
        
        # Generate random walk with trend
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, periods)
        
        # Add some trend and cycles
        trend = np.linspace(0, 0.1, periods)
        cycle = 0.05 * np.sin(np.linspace(0, 10 * np.pi, periods))
        
        prices = 100 * np.exp(np.cumsum(returns + trend / periods + cycle / periods))
        
        # Create OHLC data
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        df = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.random.uniform(0, 0.005, periods)),
            'low': prices * (1 - np.random.uniform(0, 0.005, periods)),
            'open': np.roll(prices, 1),
            'volume': np.random.randint(100, 1000, periods)
        }, index=timestamps)
        
        df['open'].iloc[0] = df['close'].iloc[0]
        
        return df
    
    def run(self, strategy, df: pd.DataFrame):
        """
        Run backtest
        
        Args:
            strategy: Trading strategy to test
            df: DataFrame with OHLC data
        """
        logger.info("=" * 60)
        logger.info("BACKTESTING STARTED")
        logger.info("=" * 60)
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Data Points: {len(df)}")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info("=" * 60)
        
        # Calculate indicators
        logger.info("Calculating technical indicators...")
        df = TechnicalIndicators.calculate_all(df)
        
        # Train ML models if using ML strategy
        if isinstance(strategy, MLStrategy):
            logger.info("Training ML models...")
            train_size = int(len(df) * 0.5)
            train_df = df.iloc[:train_size]
            strategy.train(train_df)
            df = df.iloc[train_size:]  # Test on remaining data
        
        logger.info("Running backtest...")
        
        position = None
        equity_curve = []
        
        for i in range(100, len(df)):
            current_df = df.iloc[:i]
            current_price = df.iloc[i]['close']
            
            # Record equity
            equity = self.capital
            if position:
                equity += (current_price - position['entry_price']) * position['size'] / position['entry_price']
            equity_curve.append(equity)
            
            # Get signal
            signal, confidence, metadata = strategy.get_signal(current_df, current_price)
            
            # Manage position
            if position:
                # Check exit conditions
                pnl_percent = (current_price - position['entry_price']) / position['entry_price']
                
                should_exit = False
                exit_reason = ""
                
                if position['type'] == 'BUY':
                    if pnl_percent <= -Config.STOP_LOSS_PERCENT:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif pnl_percent >= Config.TAKE_PROFIT_PERCENT:
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif signal == 'SELL' and confidence > 0.7:
                        should_exit = True
                        exit_reason = "Signal Reversal"
                else:  # SELL
                    if pnl_percent <= -Config.STOP_LOSS_PERCENT:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif pnl_percent >= Config.TAKE_PROFIT_PERCENT:
                        should_exit = True
                        exit_reason = "Take Profit"
                    elif signal == 'BUY' and confidence > 0.7:
                        should_exit = True
                        exit_reason = "Signal Reversal"
                
                if should_exit:
                    # Close position
                    pnl = position['size'] * pnl_percent
                    self.capital += pnl
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': df.index[i],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': pnl_percent * 100,
                        'exit_reason': exit_reason
                    }
                    self.trades.append(trade)
                    position = None
                    
            else:
                # Open new position
                if signal in ['BUY', 'SELL'] and confidence >= Config.MIN_CONFIDENCE_SCORE:
                    position_size = self.capital * Config.RISK_PER_TRADE
                    
                    position = {
                        'type': signal,
                        'entry_price': current_price,
                        'entry_time': df.index[i],
                        'size': position_size,
                        'confidence': confidence
                    }
        
        # Close any remaining position
        if position:
            current_price = df.iloc[-1]['close']
            pnl_percent = (current_price - position['entry_price']) / position['entry_price']
            pnl = position['size'] * pnl_percent
            self.capital += pnl
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'pnl': pnl,
                'pnl_percent': pnl_percent * 100,
                'exit_reason': 'End of backtest'
            }
            self.trades.append(trade)
        
        # Calculate metrics
        self._print_results(equity_curve)
    
    def _print_results(self, equity_curve):
        """Print backtest results"""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        if not self.trades:
            logger.warning("No trades executed")
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        winning_pnl = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        losing_pnl = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0
        
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
        
        # Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Print results
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Capital: ${self.capital:,.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Total PnL: ${total_pnl:+,.2f}")
        logger.info("")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info("")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info("=" * 60)
        
        # Sample trades
        if total_trades > 0:
            logger.info("\nSample Trades (first 5):")
            for i, trade in enumerate(self.trades[:5]):
                logger.info(f"{i+1}. {trade['type']} | "
                          f"Entry: {trade['entry_price']:.5f} | "
                          f"Exit: {trade['exit_price']:.5f} | "
                          f"PnL: ${trade['pnl']:+.2f} ({trade['pnl_percent']:+.2f}%) | "
                          f"Reason: {trade['exit_reason']}")


def main():
    """Main backtest entry point"""
    
    print("\n" + "=" * 60)
    print("DerivBot - Backtesting Module")
    print("=" * 60 + "\n")
    
    # Create backtest
    backtest = SimpleBacktest(initial_capital=10000)
    
    # Generate data
    df = backtest.generate_synthetic_data(periods=2000)
    
    print("\nSelect strategy to test:")
    print("1. Multi-Indicator Strategy")
    print("2. ML Ensemble Strategy")
    print("3. High Accuracy Strategy (60%+ Win Rate Target)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        strategy = MultiIndicatorStrategy()
    elif choice == '2':
        strategy = MLStrategy()
    elif choice == '3':
        strategy = HighAccuracyStrategy()
    else:
        print("Invalid choice, using High Accuracy Strategy")
        strategy = HighAccuracyStrategy()
    
    print()
    
    # Run backtest
    backtest.run(strategy, df)
    
    print("\n" + "=" * 60)
    print("Backtesting completed!")
    print("=" * 60)
    print("\nNote: This backtest used synthetic data.")
    print("For real backtesting, load historical data from Deriv API.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
