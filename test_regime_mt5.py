"""
Test Regime-Aware Strategy with MetaTrader 5
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 package not installed. Install with: pip install MetaTrader5")

from strategies.regime_aware_strategy import RegimeAwareStrategy, MarketRegime
from utils.regime_detector import calculate_volatility_metrics, detect_regime_simple


def connect_mt5():
    """Initialize MT5 connection"""
    if not MT5_AVAILABLE:
        return False
    
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    print(f"✓ MT5 initialized successfully")
    print(f"  Terminal: {mt5.terminal_info().name}")
    print(f"  Company: {mt5.terminal_info().company}")
    
    # Get account info
    account = mt5.account_info()
    if account:
        print(f"  Account: {account.login}")
        print(f"  Server: {account.server}")
        print(f"  Balance: ${account.balance:.2f}")
        print(f"  Leverage: 1:{account.leverage}")
    
    return True


def get_available_symbols():
    """Get synthetic indices available in MT5"""
    if not MT5_AVAILABLE:
        return []
    
    symbols = mt5.symbols_get()
    if not symbols:
        return []
    
    # Look for synthetic indices (Volatility, Crash, Boom, etc.)
    synthetic_keywords = ['Volatility', 'Vol', 'Crash', 'Boom', 'Step', 'Range', 'Jump']
    synthetic_symbols = []
    
    for sym in symbols:
        for keyword in synthetic_keywords:
            if keyword.lower() in sym.name.lower():
                synthetic_symbols.append(sym.name)
                break
    
    return synthetic_symbols


def get_ohlc_data(symbol: str, timeframe=mt5.TIMEFRAME_M5, bars: int = 200):
    """Get OHLC data from MT5"""
    if not MT5_AVAILABLE:
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to get rates for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    
    return df


def test_regime_on_symbol(symbol: str, strategy: RegimeAwareStrategy):
    """Test regime detection and signal generation on a symbol"""
    print(f"\n{'='*60}")
    print(f"TESTING: {symbol}")
    print(f"{'='*60}")
    
    # Get data
    df = get_ohlc_data(symbol, bars=200)
    if df is None:
        print(f"  ✗ Could not get data for {symbol}")
        return None
    
    print(f"  Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    current_price = df['close'].iloc[-1]
    print(f"  Current price: {current_price:.5f}")
    
    # Calculate volatility metrics
    vol_metrics = calculate_volatility_metrics(df)
    print(f"\n  Volatility Analysis:")
    print(f"    Current volatility: {vol_metrics['current_volatility']:.4f}")
    print(f"    Historical volatility: {vol_metrics['historical_volatility']:.4f}")
    print(f"    Volatility ratio: {vol_metrics['volatility_ratio']:.2f}")
    print(f"    Volatility percentile: {vol_metrics['volatility_percentile']:.1f}th")
    print(f"    ATR: {vol_metrics['atr']:.5f}")
    
    # Get simple regime
    simple_regime = detect_regime_simple(df)
    print(f"\n  Quick Regime Check: {simple_regime.value}")
    
    # Run full strategy analysis
    print(f"\n  Full Strategy Analysis:")
    try:
        signal, confidence, metadata = strategy.analyze(df, current_price)
        
        print(f"    Signal: {signal}")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Regime: {metadata.get('regime', 'N/A')}")
        print(f"    Reason: {metadata.get('reason', 'N/A')}")
        
        if 'stop_loss' in metadata and metadata['stop_loss']:
            print(f"    Stop Loss: {metadata['stop_loss']:.5f}")
            print(f"    Take Profit: {metadata['take_profit']:.5f}")
            print(f"    Position Size Mult: {metadata.get('position_size_multiplier', 1.0):.2f}x")
            print(f"    Max Hold Bars: {metadata.get('max_hold_bars', 'N/A')}")
        
        if 'volatility_percentile' in metadata:
            print(f"    Vol Percentile: {metadata['volatility_percentile']:.1f}th")
        
        # Get regime summary
        regime_summary = strategy.get_regime_summary()
        print(f"\n  Strategy State:")
        print(f"    Current Regime: {regime_summary['current_regime']}")
        print(f"    Regime Age: {regime_summary['regime_age_bars']} bars")
        print(f"    Kill Switch: {'ACTIVE - ' + str(regime_summary['kill_switch_reason']) if regime_summary['kill_switch_active'] else 'OFF'}")
        print(f"    Daily Trades: {regime_summary['daily_trades']}")
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'regime': metadata.get('regime'),
            'price': current_price
        }
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("REGIME-AWARE STRATEGY - MT5 LIVE TEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to MT5
    if not connect_mt5():
        print("\n✗ Failed to connect to MT5. Make sure MetaTrader 5 is running.")
        return False
    
    # Get available synthetic symbols
    print("\n" + "-"*60)
    print("AVAILABLE SYNTHETIC INDICES:")
    print("-"*60)
    
    symbols = get_available_symbols()
    if not symbols:
        print("No synthetic indices found. Checking all symbols...")
        all_symbols = mt5.symbols_get()
        if all_symbols:
            # Just take first 5 for testing
            symbols = [s.name for s in all_symbols[:10]]
            print(f"Using first 10 symbols for testing: {symbols}")
    else:
        print(f"Found {len(symbols)} synthetic indices:")
        for sym in symbols[:15]:  # Show first 15
            print(f"  - {sym}")
    
    # Initialize strategy
    strategy = RegimeAwareStrategy()
    print(f"\n✓ Strategy initialized: {strategy.name}")
    
    # Test on symbols
    results = []
    test_symbols = symbols[:5] if symbols else []  # Test first 5
    
    if not test_symbols:
        print("\n✗ No symbols available for testing")
        mt5.shutdown()
        return False
    
    for symbol in test_symbols:
        result = test_regime_on_symbol(symbol, strategy)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if results:
        print(f"\nAnalyzed {len(results)} symbols:")
        for r in results:
            signal_emoji = "🟢" if r['signal'] == 'BUY' else "🔴" if r['signal'] == 'SELL' else "⚪"
            print(f"  {signal_emoji} {r['symbol']}: {r['signal']} ({r['confidence']:.0%}) - {r['regime']}")
        
        # Count signals
        buys = sum(1 for r in results if r['signal'] == 'BUY')
        sells = sum(1 for r in results if r['signal'] == 'SELL')
        neutrals = sum(1 for r in results if r['signal'] == 'NEUTRAL')
        
        print(f"\nSignal distribution:")
        print(f"  BUY: {buys}, SELL: {sells}, NEUTRAL: {neutrals}")
    else:
        print("No results to display")
    
    # Cleanup
    mt5.shutdown()
    print("\n✓ MT5 connection closed")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
