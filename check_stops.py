import MetaTrader5 as mt5

if mt5.initialize():
    symbol_info = mt5.symbol_info("Volatility 100 Index")
    if symbol_info:
        print(f"\n=== R_100 Symbol Info ===")
        print(f"Symbol: {symbol_info.name}")
        print(f"Stops Level: {symbol_info.trade_stops_level}")
        print(f"Point: {symbol_info.point}")
        print(f"Digits: {symbol_info.digits}")
        print(f"Minimum Stop Distance (in price): {symbol_info.trade_stops_level * symbol_info.point}")
        print(f"Current Bid: {symbol_info.bid}")
        print(f"Current Ask: {symbol_info.ask}")
        
        # Calculate what our 0.78 stop would be
        current_price = 783.11
        our_stop = 783.89
        our_distance = abs(our_stop - current_price)
        min_required = symbol_info.trade_stops_level * symbol_info.point
        
        print(f"\n=== Trade Analysis ===")
        print(f"Entry Price: {current_price}")
        print(f"Our Stop Loss: {our_stop}")
        print(f"Our Stop Distance: {our_distance:.5f}")
        print(f"Minimum Required: {min_required:.5f}")
        print(f"Difference: {our_distance - min_required:.5f}")
        print(f"Status: {'✓ VALID' if our_distance >= min_required else '✗ TOO TIGHT'}")
    
    mt5.shutdown()
