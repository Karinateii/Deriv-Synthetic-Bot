from api.deriv_api import DerivAPI
import time
from loguru import logger

# Disable verbose logging
logger.remove()
logger.add(lambda msg: print(msg, end=''), format="{message}")

api = DerivAPI('1089', 'DtxjHvf7EsEGHPi')
api.connect()
time.sleep(2)

balance = api.get_account_balance()
print(f"\n=== ACCOUNT INFO ===")
print(f"Balance: ${balance:.2f}")
print(f"Status: {'FUNDED' if balance > 0 else 'ZERO BALANCE - Need to fund account'}")

api.close()
