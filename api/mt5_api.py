"""
MetaTrader 5 API Integration for Deriv
"""
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime
from typing import Optional, Dict, List, Callable
from loguru import logger


class MT5API:
    """Handles MetaTrader 5 connection and trading"""
    
    # Symbol mapping: Deriv API names -> MT5 names
    SYMBOL_MAP = {
        # Volatility Indices
        'R_10': 'Volatility 10 Index',
        '1HZ10V': 'Volatility 10 (1s) Index',
        'R_25': 'Volatility 25 Index',
        '1HZ25V': 'Volatility 25 (1s) Index',
        'R_50': 'Volatility 50 Index',
        '1HZ50V': 'Volatility 50 (1s) Index',
        'R_75': 'Volatility 75 Index',
        '1HZ75V': 'Volatility 75 (1s) Index',
        'R_100': 'Volatility 100 Index',
        '1HZ100V': 'Volatility 100 (1s) Index',
    }
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, 
                 server: Optional[str] = None):
        """
        Initialize MT5 connection
        
        Args:
            login: MT5 account number (optional if already logged in)
            password: MT5 account password
            server: MT5 server name (e.g., 'Deriv-Demo', 'Deriv-Real')
        """
        self.login = login
        self.password = password
        self.server = server
        self.is_connected = False
        self.authorized = False
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialize() failed: {error}")
                return False
            
            self.is_connected = True
            logger.success("Connected to MT5")
            
            # Login if credentials provided
            if self.login and self.password and self.server:
                return self.authorize()
            else:
                # Check if already logged in
                account_info = mt5.account_info()
                if account_info is None:
                    logger.warning("Not logged into any MT5 account")
                    return False
                    
                self.authorized = True
                logger.info(f"Using existing MT5 login: {account_info.login}")
                return True
                
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def authorize(self) -> bool:
        """Login to MT5 account"""
        try:
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                return False
                
            self.authorized = True
            logger.success(f"Logged into MT5 account: {self.login}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 authorization error: {e}")
            return False
            
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.0
            return float(account_info.balance)
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
            
    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
                
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'server': account_info.server,
                'currency': account_info.currency,
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            # Enable symbol in Market Watch
            if not mt5.symbol_select(mt5_symbol, True):
                logger.warning(f"Failed to select symbol: {mt5_symbol}")
                return None
                
            info = mt5.symbol_info(mt5_symbol)
            if info is None:
                return None
                
            return {
                'name': info.name,
                'bid': info.bid,
                'ask': info.ask,
                'spread': info.spread,
                'digits': info.digits,
                'trade_mode': info.trade_mode,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'point': info.point,
                'stops_level': info.trade_stops_level,  # Minimum stop distance in points
                'min_stop_distance': info.trade_stops_level * info.point,  # In price
            }
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
            
    def get_ticks_history(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Get historical tick data"""
        try:
            mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            # Get ticks from now backwards
            ticks = mt5.copy_ticks_from(mt5_symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                logger.warning(f"No tick data for {mt5_symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting tick history: {e}")
            return pd.DataFrame()
            
    def get_rates(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, count: int = 1000) -> pd.DataFrame:
        """
        Get OHLC bar data
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe (mt5.TIMEFRAME_M1, M5, M15, H1, etc.)
            count: Number of bars
        """
        try:
            mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            rates = mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No rate data for {mt5_symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates: {e}")
            return pd.DataFrame()
            
    def buy(self, symbol: str, volume: float, stop_loss: Optional[float] = None, 
            take_profit: Optional[float] = None, comment: str = "") -> Optional[Dict]:
        """
        Open BUY position
        
        Args:
            symbol: Trading symbol
            volume: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            Order result dict or None
        """
        try:
            mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            # Get current price
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info is None:
                logger.error(f"Symbol not found: {mt5_symbol}")
                return None
                
            price = symbol_info.ask
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": comment or "DerivBot BUY",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Changed from IOC to FOK for Deriv
            }
            
            # Add SL/TP if provided
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
                
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return None
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment} (code {result.retcode})")
                return None
                
            logger.success(f"BUY order placed: {mt5_symbol} | Ticket: {result.order} | Volume: {volume}")
            
            return {
                'ticket': result.order,
                'symbol': mt5_symbol,
                'type': 'BUY',
                'volume': volume,
                'price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'comment': comment,
            }
            
        except Exception as e:
            logger.error(f"Error opening BUY position: {e}")
            return None
            
    def sell(self, symbol: str, volume: float, stop_loss: Optional[float] = None, 
             take_profit: Optional[float] = None, comment: str = "") -> Optional[Dict]:
        """
        Open SELL position
        
        Args:
            symbol: Trading symbol
            volume: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
            
        Returns:
            Order result dict or None
        """
        try:
            mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            # Get current price
            symbol_info = mt5.symbol_info(mt5_symbol)
            if symbol_info is None:
                logger.error(f"Symbol not found: {mt5_symbol}")
                return None
                
            price = symbol_info.bid
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": comment or "DerivBot SELL",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Changed from IOC to FOK for Deriv
            }
            
            # Add SL/TP if provided
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
                
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return None
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment} (code {result.retcode})")
                return None
                
            logger.success(f"SELL order placed: {mt5_symbol} | Ticket: {result.order} | Volume: {volume}")
            
            return {
                'ticket': result.order,
                'symbol': mt5_symbol,
                'type': 'SELL',
                'volume': volume,
                'price': result.price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'comment': comment,
            }
            
        except Exception as e:
            logger.error(f"Error opening SELL position: {e}")
            return None
            
    def close_position(self, ticket: int) -> bool:
        """Close position by ticket"""
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                logger.warning(f"Position {ticket} not found")
                return False
                
            position = positions[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "DerivBot close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position {ticket}")
                return False
                
            logger.success(f"Position closed: {ticket}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
            
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open positions"""
        try:
            if symbol:
                mt5_symbol = self.SYMBOL_MAP.get(symbol, symbol)
                positions = mt5.positions_get(symbol=mt5_symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                return []
                
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'stop_loss': pos.sl,
                    'take_profit': pos.tp,
                    'profit': pos.profit,
                    'comment': pos.comment,
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None, 
                       take_profit: Optional[float] = None) -> bool:
        """Modify position SL/TP"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return False
                
            position = positions[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
            }
            
            if stop_loss:
                request["sl"] = stop_loss
            else:
                request["sl"] = position.sl
                
            if take_profit:
                request["tp"] = take_profit
            else:
                request["tp"] = position.tp
                
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                return False
                
            logger.info(f"Position {ticket} modified")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
            
    def close(self):
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self.is_connected = False
            self.authorized = False
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error closing MT5: {e}")
