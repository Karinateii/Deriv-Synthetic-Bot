# Regime-Aware Trading Strategy

## 🎯 What Is This Strategy?

This is a **smart trading strategy** designed specifically for synthetic indices (like Volatility 75, Boom/Crash, etc.). Unlike traditional strategies that trade based on simple indicators, this one first **understands what type of market we're in** before deciding to trade.

Think of it like a surfer who checks the ocean conditions before deciding whether to surf. A smart surfer doesn't just jump in - they check if the waves are good, if there's a rip current, if the wind is right. This strategy does the same thing for trading.

---

## 🧠 The Core Idea (In Simple Words)

### The Problem with Normal Strategies
Most trading bots fail because they:
- Trade the same way in all market conditions
- Get chopped up in sideways/volatile markets
- Don't know when to sit out
- Blow up during crazy market spikes

### Our Solution: Regime Detection
Before taking any trade, the strategy asks: **"What kind of market is this right now?"**

It classifies the market into one of these **regimes**:

| Regime | What It Means | Do We Trade? |
|--------|---------------|--------------|
| 🟢 **Quiet Range** | Low volatility, price bouncing between levels | ✅ Yes - Mean reversion trades |
| 🟢 **Trending Calm** | Clear direction with low noise | ✅ Yes - Trend following |
| 🟡 **Volatile Range** | High volatility, no direction | ⚠️ Limited - Very selective |
| 🟡 **Trending Volatile** | Direction but with big swings | ⚠️ Limited - Wide stops |
| 🔴 **Chaotic** | Extreme, unpredictable moves | ❌ Never trade |
| 🔴 **Transition** | Regime is changing | ❌ Wait it out |

---

## 🛡️ Safety Features (Kill Switches)

The strategy has automatic "emergency brakes" that stop trading when things go wrong:

1. **Volatility Spike** - If volatility suddenly jumps 3x normal → STOP
2. **Consecutive Losses** - After 3 losses in a row → STOP
3. **Daily Trade Limit** - After 10 trades per day → STOP
4. **Drawdown Limit** - If down 3% in a session → STOP
5. **Rapid Regime Changes** - If market keeps flipping regimes → STOP
6. **Price Anomaly** - If there's an extreme candle (5x normal) → STOP

---

## 💰 How Position Sizing Works

Instead of risking the same amount on every trade, the strategy adjusts based on conditions:

| Condition | Risk Adjustment |
|-----------|-----------------|
| High confidence signal | More risk (up to 1.5x) |
| Low confidence signal | Less risk (down to 0.5x) |
| Quiet regime | Normal to slightly more |
| Volatile regime | Half size or less |
| Young regime (just detected) | Reduced size |

**Base Risk:** 1% per trade  
**Range:** 0.5% to 2.5% per trade (adaptive)

---

## 📊 Entry Logic By Regime

### In Quiet Range Markets
**Strategy:** Mean Reversion (fade extremes)
- Wait for price to move 2+ standard deviations from mean
- Check that momentum is fading (slowing down)
- Enter opposite direction expecting return to mean
- Target: The mean price
- Stop: 2.5x ATR away

### In Trending Calm Markets
**Strategy:** Trend Continuation (pullback entries)
- Identify trend direction (using EMAs)
- Wait for pullback to support/resistance
- Confirm with bullish/bearish candles
- Enter with the trend
- Target: 2.5x risk distance
- Stop: Below recent swing

### In Volatile Markets
**Strategy:** Volatility Fade (very selective)
- Only trade after 3+ ATR moves
- Look for exhaustion (range declining)
- Fade the move expecting snapback
- Half position size
- Quick exits

---

## 📁 Files Overview

```
DerivBot/
├── strategies/
│   └── regime_aware_strategy.py    # Main strategy logic
├── utils/
│   └── regime_detector.py          # Helper functions
├── regime_trading_bot.py           # Automated trading bot
├── backtest_regime.py              # Test on historical data
├── test_regime_mt5.py              # Manual testing script
├── live_regime_monitor.py          # Real-time monitoring
└── config.py                       # Configuration settings
```

---

## 🚀 How To Run

### Prerequisites
1. **MetaTrader 5** installed and logged into Deriv account
2. **Python 3.10+** installed
3. Required packages installed

### Step 1: Install Dependencies
```powershell
cd DerivBot
pip install numpy pandas MetaTrader5 loguru python-dotenv
```

### Step 2: Configure (Optional)
Edit `.env` file or `config.py` to adjust:
- Risk per trade
- Max daily trades
- Symbols to monitor

### Step 3: Run the Bot
```powershell
# Make sure MT5 is open and logged in first!

cd DerivBot
python regime_trading_bot.py
```

### Step 4: Monitor
The bot will:
- Print status every 5 minutes
- Log all activity to `logs/regime_bot_YYYYMMDD.log`
- Show open positions and P&L

### To Stop
Press `Ctrl+C` in the terminal

---

## 🧪 Testing Without Real Trading

### Run Backtest (Historical Data)
```powershell
python backtest_regime.py
```

### Monitor Live Without Trading
```powershell
python live_regime_monitor.py
```

### Quick Analysis Test
```powershell
python test_regime_mt5.py
```

---

## ⚙️ Configuration Options

You can adjust these in `config.py` or via environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `REGIME_MIN_REGIME_AGE` | 3 | Bars before trading new regime |
| `REGIME_MIN_CONFIDENCE` | 0.50 | Minimum signal confidence |
| `REGIME_BASE_RISK` | 0.01 | Base risk per trade (1%) |
| `REGIME_MAX_RISK` | 0.025 | Maximum risk per trade (2.5%) |
| `REGIME_MAX_DAILY_TRADES` | 10 | Max trades per day |
| `REGIME_MAX_CONSECUTIVE_LOSSES` | 3 | Kill switch trigger |
| `REGIME_DRAWDOWN_PAUSE` | 0.03 | Stop at 3% daily loss |

---

## ⚠️ Important Disclaimers

1. **No Profit Guarantee** - This strategy is designed for capital preservation, not guaranteed profits
2. **Use Demo First** - Always test on a demo account before risking real money
3. **Synthetic Index Risk** - Broker controls price generation; be cautious
4. **Long Flat Periods** - The strategy may not trade for hours or even days - this is by design
5. **Past Performance** - Backtest results don't guarantee future performance

---

## 📈 What Success Looks Like

With this strategy, success means:
- ✅ **Small losses** when wrong (1-2% per trade max)
- ✅ **Sitting out** during dangerous conditions
- ✅ **Capital preserved** over time
- ✅ **Selective trading** (quality over quantity)
- ✅ **No blow-ups** from volatility spikes

It does NOT mean:
- ❌ Trading every day
- ❌ Huge winning streaks
- ❌ Getting rich quick

---

## 🔄 Strategy Flow

```
┌─────────────────────────────────────────────────────────┐
│                    EVERY 60 SECONDS                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              CHECK KILL SWITCH CONDITIONS                │
│   • Volatility spike? • Consecutive losses?              │
│   • Daily limit? • Drawdown?                             │
└─────────────────────────────────────────────────────────┘
                            │
                     Kill switch OFF
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   DETECT MARKET REGIME                   │
│   • Calculate volatility percentile                      │
│   • Check trend strength                                 │
│   • Classify: quiet/trending/volatile/chaotic            │
└─────────────────────────────────────────────────────────┘
                            │
                    Regime is tradeable
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 CHECK REGIME AGE                         │
│   • Is regime stable? (3+ bars)                         │
│   • Not too old? (< 200 bars)                           │
└─────────────────────────────────────────────────────────┘
                            │
                     Regime confirmed
                            ▼
┌─────────────────────────────────────────────────────────┐
│               GENERATE TRADE SETUP                       │
│   • Quiet → Mean reversion                               │
│   • Trending → Pullback continuation                     │
│   • Volatile → Fade extremes (if any)                    │
└─────────────────────────────────────────────────────────┘
                            │
                    Setup found + confidence > 50%
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   EXECUTE TRADE                          │
│   • Calculate position size (adaptive)                   │
│   • Set SL/TP based on ATR                              │
│   • Open trade on MT5                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🆘 Troubleshooting

### "MT5 initialization failed"
- Make sure MetaTrader 5 is open and logged in
- Check that you're using the right server (Deriv-Demo or Deriv-Server)

### "No trades executing"
- This is normal! The strategy is very selective
- Check the logs for regime status
- Markets may be in untradeable regime (chaotic, transition)

### "Kill switch activated"
- Check `logs/` for the reason
- Wait for conditions to improve or new trading day
- Manually reset: close and restart the bot

### "Module not found"
```powershell
pip install numpy pandas MetaTrader5 loguru python-dotenv
```

---

## 📞 Quick Commands

```powershell
# Start the bot
python regime_trading_bot.py

# Run backtest
python backtest_regime.py

# Monitor only (no trading)
python live_regime_monitor.py

# Quick test
python test_regime_mt5.py
```

---

**Happy Trading! 🚀**

*Remember: The best trade is sometimes no trade at all.*
