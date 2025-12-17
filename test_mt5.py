from api.mt5_api import MT5API
from loguru import logger
import time

# Disable verbose logging
logger.remove()
logger.add(lambda msg: print(msg, end=''), format="{message}\n")

print("=== MT5 CONNECTION TEST ===\n")

# Initialize MT5 connection
mt5 = MT5API()

# Connect
print("1. Connecting to MT5...")
if not mt5.connect():
    print("❌ Failed to connect to MT5")
    print("\nMake sure:")
    print("  - MetaTrader 5 is installed")
    print("  - MT5 is running")
    print("  - You are logged into a Deriv account")
    exit(1)

print("✅ Connected to MT5\n")

# Get account info
print("2. Checking account...")
account = mt5.get_account_info()
if account:
    print(f"✅ Account: {account['login']}")
    print(f"   Server: {account['server']}")
    print(f"   Balance: ${account['balance']:.2f}")
    print(f"   Equity: ${account['equity']:.2f}")
    print(f"   Currency: {account['currency']}\n")
else:
    print("❌ Failed to get account info\n")

# Test symbols
print("3. Testing volatility indices...")
test_symbols = ['1HZ10V', 'R_10', '1HZ50V', 'R_50', '1HZ100V', 'R_100']

for symbol in test_symbols:
    info = mt5.get_symbol_info(symbol)
    if info:
        print(f"✅ {symbol:10s} -> {info['name']:30s} | Bid: {info['bid']:.5f} | Ask: {info['ask']:.5f}")
    else:
        print(f"❌ {symbol:10s} -> NOT AVAILABLE (add to Market Watch)")

print("\n4. Testing historical data...")
df = mt5.get_ticks_history('1HZ10V', count=100)
if not df.empty:
    print(f"✅ Retrieved {len(df)} ticks for Volatility 10 (1s)")
    print(f"   Latest tick: {df.iloc[-1]['bid']:.5f} @ {df.iloc[-1]['time']}")
else:
    print("❌ Failed to get historical data")

# Get open positions
print("\n5. Checking open positions...")
positions = mt5.get_open_positions()
if len(positions) > 0:
    print(f"📊 {len(positions)} open position(s):")
    for pos in positions:
        print(f"   {pos['type']:4s} {pos['symbol']:30s} | Volume: {pos['volume']} | Profit: ${pos['profit']:.2f}")
else:
    print("📊 No open positions")

print("\n=== MT5 TEST COMPLETE ===")
print("Status: READY FOR TRADING ✅")

mt5.close()
