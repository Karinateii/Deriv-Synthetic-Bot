"""
Configuration management for DerivBot
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class"""
    
    # API Configuration
    DERIV_APP_ID = os.getenv('DERIV_APP_ID', '')
    DERIV_API_TOKEN = os.getenv('DERIV_API_TOKEN', '')
    DERIV_WS_URL = 'wss://ws.binaryws.com/websockets/v3'
    
    # MT5 Configuration
    USE_MT5 = os.getenv('USE_MT5', 'false').lower() == 'true'
    MT5_LOGIN = int(os.getenv('MT5_LOGIN', 0)) if os.getenv('MT5_LOGIN', '').strip() else None
    MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER = os.getenv('MT5_SERVER', 'Deriv-Demo')
    
    # Capital & Risk Management
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1000))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))  # 2% per trade
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.05))  # 5% daily loss limit
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.1))  # 10% max position
    
    # Trading Symbols
    TRADING_SYMBOLS = os.getenv('TRADING_SYMBOLS', 'R_10,R_25,R_50,R_75,R_100').split(',')
    
    # Strategy Configuration
    PRIMARY_STRATEGY = os.getenv('PRIMARY_STRATEGY', 'ml_ensemble')
    ENABLE_AI_PREDICTION = os.getenv('ENABLE_AI_PREDICTION', 'true').lower() == 'true'
    MIN_CONFIDENCE_SCORE = float(os.getenv('MIN_CONFIDENCE_SCORE', 0.65))
    
    # Technical Indicators
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    BB_PERIOD = 20
    BB_STD = 2
    
    EMA_SHORT = 9
    EMA_MEDIUM = 21
    EMA_LONG = 50
    
    # Risk Management
    USE_TRAILING_STOP = os.getenv('USE_TRAILING_STOP', 'true').lower() == 'true'
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
    TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 0.04))
    MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', 3))
    
    # ML Model Configuration
    LSTM_SEQUENCE_LENGTH = 60
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    
    MODEL_RETRAIN_INTERVAL = 24  # hours
    MIN_TRAINING_SAMPLES = 1000
    
    # Backtesting
    BACKTEST_START_DATE = '2023-01-01'
    BACKTEST_INITIAL_CAPITAL = 10000
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_TRADE_LOGGING = os.getenv('ENABLE_TRADE_LOGGING', 'true').lower() == 'true'
    LOG_DIR = Path('logs')
    DATA_DIR = Path('data')
    MODELS_DIR = Path('models')
    
    # Create directories
    LOG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DERIV_APP_ID:
            raise ValueError("DERIV_APP_ID is required")
        if not cls.DERIV_API_TOKEN:
            raise ValueError("DERIV_API_TOKEN is required")
        return True
