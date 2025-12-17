"""
Deriv API WebSocket Connection Handler
"""
import json
import websocket
import threading
import time
from typing import Callable, Dict, Any, Optional
from loguru import logger
from config import Config


class DerivAPI:
    """Handles WebSocket connection to Deriv API"""
    
    def __init__(self, app_id: str, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = f"{Config.DERIV_WS_URL}?app_id={app_id}"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.authorized = False
        self.callbacks: Dict[str, list] = {}
        self.request_id = 0
        self.responses: Dict[int, Any] = {}
        self.lock = threading.Lock()
        
    def connect(self):
        """Establish WebSocket connection"""
        try:
            logger.info(f"Connecting to Deriv API: {self.ws_url}")
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run WebSocket in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.is_connected:
                raise ConnectionError("Failed to connect to Deriv API")
                
            logger.success("Connected to Deriv API")
            
            # Authorize
            self.authorize()
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
            
    def authorize(self):
        """Authorize with API token"""
        try:
            logger.info("Authorizing with API token...")
            response = self.send_request({
                "authorize": self.api_token
            })
            
            if 'error' in response:
                raise Exception(f"Authorization failed: {response['error']['message']}")
                
            self.authorized = True
            logger.success("Authorization successful")
            return response
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            raise
            
    def send_request(self, request: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """Send request and wait for response"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Deriv API")
            
        with self.lock:
            self.request_id += 1
            req_id = self.request_id
            
        request['req_id'] = req_id
        
        # Send request
        self.ws.send(json.dumps(request))
        
        # Wait for response
        start_time = time.time()
        while req_id not in self.responses:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Request {req_id} timed out")
            time.sleep(0.01)
            
        response = self.responses.pop(req_id)
        return response
        
    def subscribe(self, request: Dict[str, Any], callback: Callable):
        """Subscribe to stream with callback"""
        response = self.send_request(request)
        
        if 'error' in response:
            raise Exception(f"Subscription failed: {response['error']['message']}")
            
        # Get subscription ID
        if 'subscription' in response:
            sub_id = response['subscription']['id']
            if sub_id not in self.callbacks:
                self.callbacks[sub_id] = []
            self.callbacks[sub_id].append(callback)
            logger.info(f"Subscribed with ID: {sub_id}")
            
        return response
        
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            response = self.send_request({"balance": 1})
            if 'balance' in response:
                return float(response['balance']['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
            
    def get_active_symbols(self, market: str = 'synthetic_index') -> list:
        """Get list of active trading symbols"""
        try:
            response = self.send_request({
                "active_symbols": "brief",
                "product_type": "basic"
            })
            
            if 'active_symbols' in response:
                symbols = []
                for symbol in response['active_symbols']:
                    if market == 'synthetic_index' and symbol.get('market') == 'synthetic_index':
                        symbols.append(symbol)
                return symbols
            return []
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
            
    def get_ticks_history(self, symbol: str, count: int = 1000) -> list:
        """Get historical tick data"""
        try:
            response = self.send_request({
                "ticks_history": symbol,
                "count": count,
                "end": "latest",
                "style": "ticks"
            })
            
            if 'history' in response:
                return response['history']
            return []
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
            
    def subscribe_ticks(self, symbol: str, callback: Callable):
        """Subscribe to real-time tick stream"""
        return self.subscribe(
            {"ticks": symbol, "subscribe": 1},
            callback
        )
        
    def buy_contract(self, contract_type: str, symbol: str, amount: float, 
                     duration: int = 5, duration_unit: str = 'ticks',
                     barrier: Optional[str] = None) -> Dict:
        """
        Buy a contract
        
        Args:
            contract_type: CALL, PUT, DIGITDIFF, etc.
            symbol: Trading symbol (e.g., R_10)
            amount: Stake amount
            duration: Contract duration
            duration_unit: 'ticks', 'm' (minutes), 'h' (hours), 'd' (days)
            barrier: Optional barrier for some contract types
        """
        try:
            proposal = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "symbol": symbol
            }
            
            if barrier:
                proposal['barrier'] = barrier
                
            # Get proposal
            response = self.send_request(proposal)
            
            if 'error' in response:
                logger.error(f"Proposal error: {response['error']['message']}")
                return {'error': response['error']}
                
            if 'proposal' in response:
                proposal_id = response['proposal']['id']
                
                # Buy contract
                buy_response = self.send_request({
                    "buy": proposal_id,
                    "price": amount
                })
                
                if 'buy' in buy_response:
                    logger.success(f"Contract purchased: {buy_response['buy']['contract_id']}")
                    return buy_response['buy']
                elif 'error' in buy_response:
                    logger.error(f"Buy error: {buy_response['error']['message']}")
                    return {'error': buy_response['error']}
                    
        except Exception as e:
            logger.error(f"Error buying contract: {e}")
            return {'error': str(e)}
            
    def sell_contract(self, contract_id: int, price: float) -> Dict:
        """Sell a contract"""
        try:
            response = self.send_request({
                "sell": contract_id,
                "price": price
            })
            
            if 'sell' in response:
                logger.success(f"Contract sold: {contract_id}")
                return response['sell']
            elif 'error' in response:
                logger.error(f"Sell error: {response['error']['message']}")
                return {'error': response['error']}
                
        except Exception as e:
            logger.error(f"Error selling contract: {e}")
            return {'error': str(e)}
            
    def get_open_positions(self) -> list:
        """Get all open positions"""
        try:
            response = self.send_request({"portfolio": 1})
            
            if 'portfolio' in response:
                return response['portfolio']['contracts']
            return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    def _on_open(self, ws):
        """WebSocket opened"""
        self.is_connected = True
        logger.info("WebSocket connection opened")
        
    def _on_message(self, ws, message):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            
            # Handle response with req_id
            if 'req_id' in data:
                req_id = data['req_id']
                self.responses[req_id] = data
                
            # Handle subscription updates
            if 'subscription' in data:
                sub_id = data['subscription']['id']
                if sub_id in self.callbacks:
                    for callback in self.callbacks[sub_id]:
                        callback(data)
                        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
        self.authorized = False
        logger.warning("WebSocket connection closed")
        
        # Attempt reconnection
        logger.info("Attempting to reconnect...")
        time.sleep(5)
        try:
            self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            logger.info("Connection closed")
