"""Quick API connection test"""
import sys
from config import Config

print("=" * 50)
print("Testing Deriv API Connection")
print("=" * 50)

try:
    from api.deriv_api import DerivAPI
    
    print(f"\nApp ID: {Config.DERIV_APP_ID}")
    print("Token: " + Config.DERIV_API_TOKEN[:10] + "...")
    print("\nConnecting to Deriv...")
    
    api = DerivAPI(Config.DERIV_APP_ID, Config.DERIV_API_TOKEN)
    api.connect()
    
    print("✓ Connected successfully!")
    
    # Get balance
    balance = api.get_account_balance()
    print(f"\n💰 Account Balance: ${balance:.2f}")
    
    # Get available symbols
    print("\n📊 Available symbols...")
    symbols = api.get_active_symbols()
    print(f"✓ Found {len(symbols)} active symbols")
    
    api.close()
    print("\n✓ Connection test passed!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
