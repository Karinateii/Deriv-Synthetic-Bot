"""
Backtest for Regime-Aware Strategy
Tests on historical MT5 data
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 not installed")
    sys.exit(1)

from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    stop_loss: float
    take_profit: float
    pnl_pips: float
    pnl_percent: float
    exit_reason: str
    regime: str


class Backtester:
    """Backtest the regime-aware strategy"""
    
    def __init__(self, symbol: str, initial_balance: float = 10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.strategy = RegimeAwareStrategy()
        self.trades: List[Trade] = []
        self.equity_curve = []
        
    def get_historical_data(self, days: int = 30, timeframe=mt5.TIMEFRAME_M5) -> pd.DataFrame:
        """Fetch historical data from MT5"""
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return None
        
        # Calculate bars needed
        bars_per_day = {
            mt5.TIMEFRAME_M1: 1440,
            mt5.TIMEFRAME_M5: 288,
            mt5.TIMEFRAME_M15: 96,
            mt5.TIMEFRAME_H1: 24,
        }
        
        total_bars = days * bars_per_day.get(timeframe, 288)
        
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, total_bars)
        
        if rates is None or len(rates) == 0:
            print(f"Failed to get data for {self.symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df
    
    def run_backtest(self, df: pd.DataFrame, warmup_bars: int = 200) -> dict:
        """Run the backtest"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {self.symbol}")
        print(f"{'='*60}")
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Total bars: {len(df)}")
        print(f"Initial balance: ${self.initial_balance:,.2f}")
        print()
        
        # Reset strategy state
        self.strategy = RegimeAwareStrategy()
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [(df.index[warmup_bars], self.initial_balance)]
        
        # Current position
        in_position = False
        position_direction = None
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        entry_regime = None
        bars_in_trade = 0
        max_hold_bars = 50
        
        # Walk forward through data
        for i in range(warmup_bars, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['close']
            high = current_bar['high']
            low = current_bar['low']
            
            # Get historical window
            window = df.iloc[max(0, i-200):i+1].copy()
            
            # Check exits if in position
            if in_position:
                bars_in_trade += 1
                exit_reason = None
                exit_price = None
                
                # Check stop loss
                if position_direction == 'BUY' and low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                elif position_direction == 'SELL' and high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                
                # Check take profit
                elif position_direction == 'BUY' and high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                elif position_direction == 'SELL' and low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                
                # Check time-based exit
                elif bars_in_trade >= max_hold_bars:
                    exit_price = current_price
                    exit_reason = 'time_exit'
                
                # Execute exit
                if exit_reason:
                    # Calculate P&L
                    if position_direction == 'BUY':
                        pnl_pips = exit_price - entry_price
                    else:
                        pnl_pips = entry_price - exit_price
                    
                    pnl_percent = (pnl_pips / entry_price) * 100
                    
                    # Update balance (using 1% risk per trade)
                    risk_amount = self.balance * 0.01
                    if pnl_pips > 0:
                        # Winner: gain based on R:R
                        rr_ratio = abs(pnl_pips) / abs(entry_price - stop_loss) if entry_price != stop_loss else 1
                        profit = risk_amount * rr_ratio
                        self.balance += profit
                        self.strategy.record_trade_result(profit, is_win=True)
                    else:
                        # Loser: lose risk amount
                        self.balance -= risk_amount
                        self.strategy.record_trade_result(-risk_amount, is_win=False)
                    
                    # Record trade
                    trade = Trade(
                        symbol=self.symbol,
                        direction=position_direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=current_time,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pnl_pips=pnl_pips,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_reason,
                        regime=entry_regime
                    )
                    self.trades.append(trade)
                    
                    # Reset position
                    in_position = False
                    position_direction = None
                    bars_in_trade = 0
                    
                    # Record equity
                    self.equity_curve.append((current_time, self.balance))
            
            # Check for new entries (only if not in position)
            if not in_position:
                signal, confidence, metadata = self.strategy.analyze(window, current_price)
                
                if signal in ['BUY', 'SELL'] and confidence >= 0.55:
                    # Enter position
                    in_position = True
                    position_direction = signal
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = metadata.get('stop_loss', 0)
                    take_profit = metadata.get('take_profit', 0)
                    max_hold_bars = metadata.get('max_hold_bars', 50)
                    entry_regime = metadata.get('regime', 'unknown')
                    bars_in_trade = 0
                    
                    # Validate stop/tp
                    if stop_loss == 0 or take_profit == 0:
                        # Use default ATR-based stops
                        atr = self._calculate_atr(window)
                        if signal == 'BUY':
                            stop_loss = entry_price - (atr * 2.5)
                            take_profit = entry_price + (atr * 3.5)
                        else:
                            stop_loss = entry_price + (atr * 2.5)
                            take_profit = entry_price - (atr * 3.5)
        
        # Close any open position at end
        if in_position:
            exit_price = df['close'].iloc[-1]
            if position_direction == 'BUY':
                pnl_pips = exit_price - entry_price
            else:
                pnl_pips = entry_price - exit_price
            
            pnl_percent = (pnl_pips / entry_price) * 100
            
            trade = Trade(
                symbol=self.symbol,
                direction=position_direction,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=df.index[-1],
                stop_loss=stop_loss,
                take_profit=take_profit,
                pnl_pips=pnl_pips,
                pnl_percent=pnl_percent,
                exit_reason='end_of_data',
                regime=entry_regime
            )
            self.trades.append(trade)
        
        return self._calculate_stats()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high = df['high'].tail(period + 1)
        low = df['low'].tail(period + 1)
        close = df['close'].tail(period + 1)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    
    def _calculate_stats(self) -> dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        winners = [t for t in self.trades if t.pnl_pips > 0]
        losers = [t for t in self.trades if t.pnl_pips <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = sum(t.pnl_pips for t in winners)
        gross_loss = abs(sum(t.pnl_pips for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        # Calculate max drawdown from equity curve
        equity_values = [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        # Regime breakdown
        regime_stats = {}
        for t in self.trades:
            if t.regime not in regime_stats:
                regime_stats[t.regime] = {'wins': 0, 'losses': 0}
            if t.pnl_pips > 0:
                regime_stats[t.regime]['wins'] += 1
            else:
                regime_stats[t.regime]['losses'] += 1
        
        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_balance': self.balance,
            'max_drawdown': max_dd,
            'exit_reasons': exit_reasons,
            'regime_stats': regime_stats,
            'avg_winner': np.mean([t.pnl_percent for t in winners]) if winners else 0,
            'avg_loser': np.mean([t.pnl_percent for t in losers]) if losers else 0,
        }
    
    def print_results(self, stats: dict):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total Trades: {stats['total_trades']}")
        
        if stats['total_trades'] == 0:
            print("   No trades executed - strategy was too conservative or no setups found.")
            print("   This is expected for a capital-preservation focused strategy.")
            return
            
        print(f"   Winners: {stats['winners']} | Losers: {stats['losers']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        print(f"   Total Return: {stats['total_return']:.2f}%")
        print(f"   Final Balance: ${stats['final_balance']:,.2f}")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2f}%")
        
        if stats['total_trades'] > 0:
            print(f"   Avg Winner: {stats['avg_winner']:.3f}%")
            print(f"   Avg Loser: {stats['avg_loser']:.3f}%")
        
        print(f"\n📈 EXIT REASONS:")
        for reason, count in stats.get('exit_reasons', {}).items():
            pct = count / stats['total_trades'] * 100 if stats['total_trades'] > 0 else 0
            print(f"   {reason}: {count} ({pct:.1f}%)")
        
        print(f"\n🎯 BY REGIME:")
        for regime, data in stats.get('regime_stats', {}).items():
            total = data['wins'] + data['losses']
            wr = data['wins'] / total * 100 if total > 0 else 0
            print(f"   {regime}: {total} trades, {wr:.1f}% win rate")
        
        print("\n" + "="*60)


def main():
    """Run backtests on multiple symbols"""
    print("\n" + "="*60)
    print("REGIME-AWARE STRATEGY BACKTEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 failed: {mt5.last_error()}")
        return
    
    print(f"✓ MT5 Connected")
    print(f"  Account: {mt5.account_info().login}")
    
    # Symbols to test
    symbols = [
        'Volatility 50 Index',
        'Volatility 75 Index',
        'Volatility 100 Index',
    ]
    
    all_stats = []
    
    for symbol in symbols:
        bt = Backtester(symbol, initial_balance=10000)
        
        # Get 14 days of M5 data
        df = bt.get_historical_data(days=14, timeframe=mt5.TIMEFRAME_M5)
        
        if df is not None and len(df) > 300:
            stats = bt.run_backtest(df)
            bt.print_results(stats)
            all_stats.append((symbol, stats))
            
            # Print some sample trades
            if bt.trades:
                print(f"\n📝 SAMPLE TRADES ({min(5, len(bt.trades))} of {len(bt.trades)}):")
                for trade in bt.trades[:5]:
                    emoji = "✅" if trade.pnl_pips > 0 else "❌"
                    print(f"   {emoji} {trade.direction} @ {trade.entry_price:.2f} → {trade.exit_price:.2f}")
                    print(f"      {trade.entry_time.strftime('%m/%d %H:%M')} - {trade.exit_time.strftime('%m/%d %H:%M')}")
                    print(f"      P&L: {trade.pnl_percent:.3f}% | Exit: {trade.exit_reason} | Regime: {trade.regime}")
        else:
            print(f"✗ No data for {symbol}")
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    total_trades = sum(s['total_trades'] for _, s in all_stats)
    avg_win_rate = np.mean([s['win_rate'] for _, s in all_stats if s['total_trades'] > 0])
    avg_return = np.mean([s['total_return'] for _, s in all_stats])
    
    print(f"Total trades across all symbols: {total_trades}")
    print(f"Average win rate: {avg_win_rate:.1f}%")
    print(f"Average return: {avg_return:.2f}%")
    
    mt5.shutdown()
    print("\n✓ Backtest complete")


if __name__ == "__main__":
    main()
