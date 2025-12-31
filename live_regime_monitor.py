"""
Live Regime Monitor - Continuous monitoring of synthetic indices
Keeps running to build up regime age and show actual trading signals
"""
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 not installed")
    sys.exit(1)

from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime
from utils.regime_detector import calculate_volatility_metrics


# Symbols to monitor
SYMBOLS = [
    'Volatility 10 Index',
    'Volatility 25 Index', 
    'Volatility 50 Index',
    'Volatility 75 Index',
    'Volatility 100 Index',
]

# One strategy instance per symbol to maintain state
strategies = {}


def get_ohlc_data(symbol: str, timeframe=mt5.TIMEFRAME_M1, bars: int = 200):
    """Get OHLC data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    return df


def analyze_symbol(symbol: str):
    """Analyze a single symbol"""
    # Get or create strategy for this symbol
    if symbol not in strategies:
        strategies[symbol] = RegimeAwareStrategy()
    
    strategy = strategies[symbol]
    
    # Get data
    df = get_ohlc_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=200)
    if df is None:
        return None
    
    current_price = df['close'].iloc[-1]
    
    # Analyze
    signal, confidence, metadata = strategy.analyze(df, current_price)
    
    return {
        'symbol': symbol,
        'price': current_price,
        'signal': signal,
        'confidence': confidence,
        'regime': metadata.get('regime', 'unknown'),
        'regime_age': strategy.regime_bar_count,
        'reason': metadata.get('reason', ''),
        'stop_loss': metadata.get('stop_loss'),
        'take_profit': metadata.get('take_profit'),
        'vol_pct': metadata.get('volatility_percentile', 0),
        'kill_switch': strategy.kill_switch_active
    }


def print_dashboard(results):
    """Print a nice dashboard"""
    # Clear screen
    print("\033[2J\033[H", end="")
    
    print("=" * 80)
    print(f"  REGIME-AWARE STRATEGY - LIVE MONITOR  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Table header
    print(f"{'Symbol':<25} {'Price':>12} {'Signal':>8} {'Conf':>6} {'Regime':<18} {'Age':>4} {'Vol%':>5}")
    print("-" * 80)
    
    for r in results:
        if r is None:
            continue
            
        # Color codes
        if r['signal'] == 'BUY':
            signal_color = '\033[92m'  # Green
        elif r['signal'] == 'SELL':
            signal_color = '\033[91m'  # Red
        else:
            signal_color = '\033[90m'  # Gray
        
        reset = '\033[0m'
        
        print(f"{r['symbol']:<25} {r['price']:>12.2f} {signal_color}{r['signal']:>8}{reset} "
              f"{r['confidence']:>5.0%} {r['regime']:<18} {r['regime_age']:>4} {r['vol_pct']:>4.0f}%")
    
    print("-" * 80)
    print()
    
    # Show details for any non-neutral signals
    active_signals = [r for r in results if r and r['signal'] != 'NEUTRAL']
    
    if active_signals:
        print("📢 ACTIVE SIGNALS:")
        for r in active_signals:
            emoji = "🟢" if r['signal'] == 'BUY' else "🔴"
            print(f"\n{emoji} {r['symbol']} - {r['signal']} @ {r['confidence']:.0%}")
            print(f"   Entry: {r['price']:.5f}")
            if r['stop_loss']:
                print(f"   Stop Loss: {r['stop_loss']:.5f}")
                print(f"   Take Profit: {r['take_profit']:.5f}")
            print(f"   Reason: {r['reason']}")
    else:
        print("⏳ Waiting for trade setups... (Regime age needs to reach 8+ bars)")
        print("   The strategy is correctly being conservative on startup.")
    
    print()
    print("Press Ctrl+C to stop monitoring")


def main():
    """Main monitoring loop"""
    print("Connecting to MT5...")
    
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    print(f"✓ Connected to MT5")
    print(f"  Account: {mt5.account_info().login}")
    print(f"  Balance: ${mt5.account_info().balance:.2f}")
    print(f"\nStarting live monitor for {len(SYMBOLS)} symbols...")
    print("Building up regime age (this takes a few iterations)...\n")
    
    time.sleep(2)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Analyze all symbols
            results = []
            for symbol in SYMBOLS:
                result = analyze_symbol(symbol)
                results.append(result)
            
            # Print dashboard
            print_dashboard(results)
            print(f"\nIteration {iteration} | Refreshing in 10 seconds...")
            
            # Wait before next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    finally:
        mt5.shutdown()
        print("✓ MT5 connection closed")


if __name__ == "__main__":
    main()
