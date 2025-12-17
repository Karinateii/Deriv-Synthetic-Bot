# Deriv Synthetic Bot

A professional-grade AI-powered trading bot designed for Deriv synthetic indices. This project demonstrates advanced full-stack software engineering with multi-file architecture, robust testing, deterministic workflows, and production-ready containerization.

## 🎯 Project Overview

This bot combines multiple machine learning models, advanced trading strategies, and strict risk management to create a sophisticated trading system. Built with clean architecture principles, comprehensive logging, and Docker containerization for reproducibility.

### Key Features

- **Multiple AI/ML Models**: LSTM neural networks, ensemble learning (XGBoost, LightGBM, Random Forest)
- **Advanced Trading Strategies**: ML ensemble, multi-indicator fusion, pattern recognition, mean reversion
- **Robust Risk Management**: Position sizing (Kelly Criterion), dynamic stop loss/take profit, drawdown protection
- **Real-time Trading**: WebSocket integration for live market data, auto-reconnection, position monitoring
- **Comprehensive Testing**: Backtesting engine with performance metrics (Sharpe ratio, win rate, profit factor)
- **Production Ready**: Structured logging, error handling, monitoring, and Docker support

## 🏗️ Architecture

```
DerivBot/
├── api/                    # API integrations (Deriv, MT5)
│   ├── deriv_api.py       # WebSocket communication
│   ├── market_data.py     # Real-time data handling
│   └── mt5_api.py         # MetaTrader 5 integration
├── strategies/            # Trading strategy implementations
│   ├── base_strategy.py   # Abstract base class
│   ├── ml_strategy.py     # ML-powered strategies
│   ├── price_action.py    # Price action analysis
│   └── multi_indicator.py # Multi-indicator fusion
├── ml/                    # Machine learning models
│   ├── lstm_model.py      # LSTM neural network
│   ├── ensemble.py        # Ensemble learning models
│   └── indicators.py      # Technical indicators
├── risk/                  # Risk management module
│   └── risk_manager.py    # Position sizing, stops, limits
├── utils/                 # Utility functions
│   └── indicators.py      # Technical indicator calculations
├── main.py               # Entry point
├── config.py             # Configuration management
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Deriv account with API access
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/Karinateii/Deriv-Synthetic-Bot.git
cd Deriv-Synthetic-Bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Add your Deriv credentials to `.env`**:
   ```
   DERIV_APP_ID=your_app_id
   DERIV_API_TOKEN=your_api_token
   ```

3. **Adjust trading parameters**:
   ```
   INITIAL_CAPITAL=1000
   RISK_PER_TRADE=0.02
   PRIMARY_STRATEGY=ml_ensemble
   ```

### Running the Bot

```bash
# Run trading bot
python main.py

# Run backtesting
python backtest.py

# Test connection
python test_connection.py
```

### Docker Deployment

```bash
# Build image
docker build -t deriv-bot .

# Run container
docker run -e DERIV_APP_ID=your_id -e DERIV_API_TOKEN=your_token deriv-bot
```

## 📊 Trading Strategies

### 1. ML Ensemble Strategy
Combines multiple machine learning models (LSTM, XGBoost, LightGBM, Random Forest) for robust predictions. Uses confidence scoring to filter trades.

### 2. Multi-Indicator Strategy
Fuses 10+ technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, EMA) with signal validation.

### 3. Price Action Strategy
Identifies support/resistance levels, trend lines, and chart patterns automatically.

## 🛡️ Risk Management

- **Position Sizing**: Kelly Criterion and fixed percentage methods
- **Stop Loss & Take Profit**: Dynamic adjustment based on volatility
- **Drawdown Protection**: Daily loss limits prevent catastrophic failures
- **Correlation Analysis**: Avoids correlated position overlap
- **Maximum Position Limits**: Strict exposure controls

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Connection tests
python test_connection.py

# Balance verification
python test_balance.py

# MT5 integration tests
python test_mt5.py
```

## 📈 Performance Metrics

The bot tracks:
- Win rate and profit factor
- Sharpe ratio and Sortino ratio
- Maximum drawdown
- Return on investment (ROI)
- Risk-adjusted returns

## 🔒 Security & Best Practices

- **Environment Variables**: API credentials never hardcoded
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed audit trail of all operations
- **Validation**: Input sanitization and boundary checks
- **Graceful Shutdown**: Proper cleanup and resource management

## 📋 Configuration Options

See `.env.example` for all available configuration parameters:
- API credentials
- Trading capital and risk parameters
- Strategy selection
- Symbol selection
- Technical indicator settings
- Logging levels

## 🤝 Contributing

Improvements welcome! Key areas:
- Strategy optimization
- Model refinement
- Additional technical indicators
- Performance improvements
- Documentation enhancements

## ⚠️ Disclaimer

**Trading involves significant risk**. This bot is provided for educational purposes. Always:
- Test strategies thoroughly on demo accounts first
- Start with minimal capital
- Monitor bot performance regularly
- Never risk capital you can't afford to lose

## 📝 License

MIT License - See LICENSE for details

## 👤 Author

Ebenezer Doutimiwei - Full-Stack Software Engineer
- Portfolio: [https://portfolio-website-lzs4.vercel.app/](https://portfolio-website-lzs4.vercel.app/)
- GitHub: [https://github.com/Karinateii](https://github.com/Karinateii)
- LinkedIn: [Ebenezer Doutimiwei](https://linkedin.com/in/ebenezer-doutimiwei-b929a6208/)
