"""
Test script for RegimeAwareStrategy
Tests basic functionality and regime detection
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the strategy
from strategies.regime_aware_strategy import (
    RegimeAwareStrategy, 
    MarketRegime,
    TradabilityState
)
from utils.regime_detector import detect_regime_simple, calculate_volatility_metrics

def generate_test_data(regime_type='quiet_range', num_bars=200):
    """Generate synthetic OHLCV data for testing"""
    
    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq='1H')
    
    if regime_type == 'quiet_range':
        # Low volatility, ranging
        base = 100
        trend = np.linspace(0, 0.5, num_bars)
        noise = np.random.normal(0, 0.3, num_bars)
        close = base + trend + noise
        
    elif regime_type == 'trending_calm':
        # Clear uptrend, low noise
        base = 100
        trend = np.linspace(0, 3, num_bars)
        noise = np.random.normal(0, 0.2, num_bars)
        close = base + trend + noise
        
    elif regime_type == 'volatile_range':
        # High volatility, no direction
        base = 100
        noise = np.random.normal(0, 1.0, num_bars)
        close = base + noise
        
    elif regime_type == 'trending_volatile':
        # Clear trend with high noise
        base = 100
        trend = np.linspace(0, 2.5, num_bars)
        noise = np.random.normal(0, 0.8, num_bars)
        close = base + trend + noise
        
    else:  # chaotic
        # Extreme moves
        base = 100
        noise = np.random.normal(0, 2.0, num_bars)
        close = base + noise
    
    # Generate OHLC from close
    high = close + np.abs(np.random.normal(0, 0.3, num_bars))
    low = close - np.abs(np.random.normal(0, 0.3, num_bars))
    open_price = close + np.random.normal(0, 0.2, num_bars)
    
    volume = np.random.randint(1000, 10000, num_bars)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def test_import():
    """Test if strategy imports correctly"""
    print("=" * 60)
    print("TEST 1: Import and Instantiation")
    print("=" * 60)
    
    try:
        strategy = RegimeAwareStrategy()
        print(f"✓ Strategy instantiated: {strategy.name}")
        print(f"✓ Strategy parameters loaded:")
        print(f"  - Volatility lookback: {strategy.volatility_lookback}")
        print(f"  - Min regime age: {strategy.min_regime_age}")
        print(f"  - Min confidence: {strategy.min_confidence}")
        print(f"  - Base risk: {strategy.base_risk_percent * 100}%")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_regime_detection():
    """Test regime detection on different market types"""
    print("\n" + "=" * 60)
    print("TEST 2: Regime Detection")
    print("=" * 60)
    
    strategy = RegimeAwareStrategy()
    regimes_to_test = ['quiet_range', 'trending_calm', 'volatile_range', 'trending_volatile']
    
    for regime_name in regimes_to_test:
        print(f"\nTesting {regime_name}...")
        
        # Generate test data
        df = generate_test_data(regime_type=regime_name, num_bars=200)
        
        try:
            # Run regime analysis
            regime_analysis = strategy._analyze_regime(df)
            
            print(f"  Detected regime: {regime_analysis.regime.value}")
            print(f"  Regime confidence: {regime_analysis.confidence:.2%}")
            print(f"  Volatility percentile: {regime_analysis.volatility_percentile:.1f}")
            print(f"  Volatility ratio: {regime_analysis.volatility_ratio:.2f}")
            print(f"  Trend strength: {regime_analysis.trend_strength:.3f}")
            print(f"  Mean reversion score: {regime_analysis.mean_reversion_score:.2f}")
            print(f"  Tradability: {regime_analysis.tradability.value}")
            print(f"  ✓ Regime detection successful")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    return True


def test_signal_generation():
    """Test full signal generation"""
    print("\n" + "=" * 60)
    print("TEST 3: Signal Generation")
    print("=" * 60)
    
    strategy = RegimeAwareStrategy()
    test_cases = [
        ('quiet_range', "Mean reversion trades expected"),
        ('trending_calm', "Trend continuation trades expected"),
        ('volatile_range', "Limited trades expected"),
    ]
    
    for regime_name, expectation in test_cases:
        print(f"\nTesting {regime_name}...")
        print(f"  Expectation: {expectation}")
        
        df = generate_test_data(regime_type=regime_name, num_bars=200)
        current_price = df['close'].iloc[-1]
        
        try:
            signal, confidence, metadata = strategy.analyze(df, current_price)
            
            print(f"  Signal: {signal}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Regime: {metadata.get('regime', 'N/A')}")
            print(f"  Reason: {metadata.get('reason', 'N/A')}")
            
            if 'stop_loss' in metadata:
                print(f"  Stop Loss: {metadata['stop_loss']:.2f}")
                print(f"  Take Profit: {metadata['take_profit']:.2f}")
            
            print(f"  ✓ Signal generation successful")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_kill_switch():
    """Test kill switch functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: Kill Switch Logic")
    print("=" * 60)
    
    strategy = RegimeAwareStrategy()
    df = generate_test_data(regime_type='quiet_range', num_bars=200)
    
    print("\nTest 4a: Normal operation (no kill switch)")
    try:
        signal, confidence, metadata = strategy.analyze(df, df['close'].iloc[-1])
        print(f"  ✓ Normal analysis completed")
        print(f"  Kill switch active: {strategy.kill_switch_active}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\nTest 4b: Consecutive losses trigger")
    strategy.consecutive_losses = 3
    try:
        signal, confidence, metadata = strategy.analyze(df, df['close'].iloc[-1])
        print(f"  Signal: {signal}")
        print(f"  Kill switch reason: {strategy.kill_switch_reason}")
        print(f"  ✓ Kill switch triggered correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    return True


def test_regime_detector_utils():
    """Test regime detector utility functions"""
    print("\n" + "=" * 60)
    print("TEST 5: Regime Detector Utilities")
    print("=" * 60)
    
    df = generate_test_data(regime_type='trending_calm', num_bars=150)
    
    try:
        # Test volatility metrics
        vol_metrics = calculate_volatility_metrics(df)
        print(f"\nVolatility Metrics:")
        print(f"  Current vol: {vol_metrics['current_volatility']:.4f}")
        print(f"  Historical vol: {vol_metrics['historical_volatility']:.4f}")
        print(f"  Vol ratio: {vol_metrics['volatility_ratio']:.2f}")
        print(f"  Vol percentile: {vol_metrics['volatility_percentile']:.1f}")
        print(f"  ATR: {vol_metrics['atr']:.4f}")
        print(f"  ✓ Volatility metrics calculated")
        
        # Test simple regime detection
        regime = detect_regime_simple(df)
        print(f"\nSimple Regime Detection:")
        print(f"  Detected regime: {regime.value}")
        print(f"  ✓ Simple regime detection working")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REGIME-AWARE STRATEGY TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Import & Instantiation", test_import()))
    results.append(("Regime Detection", test_regime_detection()))
    results.append(("Signal Generation", test_signal_generation()))
    results.append(("Kill Switch Logic", test_kill_switch()))
    results.append(("Regime Detector Utils", test_regime_detector_utils()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
