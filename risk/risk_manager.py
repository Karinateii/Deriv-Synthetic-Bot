"""
Risk Management System
"""
import numpy as np
from typing import Optional, Tuple
from loguru import logger
from config import Config


class RiskManager:
    """Manages risk for trading operations"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.open_positions = 0
        self.daily_trades = 0
        
        self.is_trading_enabled = True
        
        # Track trade history for better statistics
        self.trade_history = []
        self.win_amounts = []
        self.loss_amounts = []
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                               method: str = 'fixed_percent') -> float:
        """
        Calculate position size based on risk
        
        Args:
            entry_price: Entry price for trade
            stop_loss_price: Stop loss price
            method: 'fixed_percent', 'kelly', or 'fixed_amount'
            
        Returns:
            Position size (stake amount)
        """
        if method == 'fixed_percent':
            # Risk fixed percentage per trade
            risk_amount = self.current_capital * Config.RISK_PER_TRADE
            
            # Calculate position size based on stop loss distance
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                position_size = risk_amount
            
            # Ensure within limits
            max_position = self.current_capital * Config.MAX_POSITION_SIZE
            position_size = min(position_size, max_position)
            
            return position_size
            
        elif method == 'kelly':
            # Kelly Criterion
            win_rate = self.get_win_rate()
            
            if win_rate > 0.5 and self.total_trades >= 10:
                avg_win = self.get_average_win()
                avg_loss = self.get_average_loss()
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                    
                    position_size = self.current_capital * kelly_fraction
                    return position_size
            
            # Fallback to fixed percent
            return self.current_capital * Config.RISK_PER_TRADE
            
        elif method == 'fixed_amount':
            return self.current_capital * Config.RISK_PER_TRADE
        
        return self.current_capital * Config.RISK_PER_TRADE
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr: Optional[float] = None, use_dynamic: bool = True) -> float:
        """
        Calculate stop loss price with improved ATR-based logic
        
        Args:
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            atr: Average True Range (optional, for dynamic stops)
            use_dynamic: Use dynamic ATR-based stops (recommended)
            
        Returns:
            Stop loss price
        """
        if atr and atr > 0 and use_dynamic:
            # Dynamic ATR-based stop (2x ATR for breathing room)
            stop_distance = atr * 2.0
            # But ensure minimum risk
            min_stop = entry_price * 0.015  # Min 1.5% stop
            stop_distance = max(stop_distance, min_stop)
        else:
            # Fixed percentage stop - reduced from 5% to 2.5% for tighter control
            stop_distance = entry_price * 0.025
        
        if direction == 'BUY':
            stop_loss = entry_price - stop_distance
        else:  # SELL
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             direction: str, risk_reward_ratio: float = 2.5) -> float:
        """
        Calculate take profit price with improved risk:reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price (to calculate actual risk)
            direction: 'BUY' or 'SELL'
            risk_reward_ratio: Risk:Reward ratio (default 2.5:1 for better profits)
            
        Returns:
            Take profit price
        """
        # Calculate actual stop distance
        stop_distance = abs(entry_price - stop_loss)
        profit_distance = stop_distance * risk_reward_ratio
        
        if direction == 'BUY':
            take_profit = entry_price + profit_distance
        else:  # SELL
            take_profit = entry_price - profit_distance
        
        return take_profit
    
    def can_open_position(self, stake_amount: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened
        
        Returns:
            (can_open, reason)
        """
        # Check if trading is enabled
        if not self.is_trading_enabled:
            return False, "Trading is disabled"
        
        # Check maximum positions
        if self.open_positions >= Config.MAX_OPEN_POSITIONS:
            return False, f"Maximum positions reached ({Config.MAX_OPEN_POSITIONS})"
        
        # Check capital availability
        if stake_amount > self.current_capital * Config.MAX_POSITION_SIZE:
            return False, f"Position size exceeds maximum ({Config.MAX_POSITION_SIZE * 100}%)"
        
        if stake_amount > self.current_capital:
            return False, "Insufficient capital"
        
        # Check daily loss limit
        daily_loss_percent = abs(self.daily_pnl / self.initial_capital)
        if self.daily_pnl < 0 and daily_loss_percent > Config.MAX_DAILY_LOSS:
            self.is_trading_enabled = False
            return False, f"Daily loss limit reached ({Config.MAX_DAILY_LOSS * 100}%)"
        
        # Check drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > Config.MAX_DAILY_LOSS * 2:  # 2x daily loss as max drawdown
            self.is_trading_enabled = False
            return False, f"Maximum drawdown reached"
        
        return True, "OK"
    
    def record_trade(self, pnl: float, is_win: bool):
        """Record trade result with enhanced tracking"""
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        
        # Track win/loss amounts for better statistics
        if is_win:
            self.winning_trades += 1
            self.win_amounts.append(abs(pnl))
        else:
            self.losing_trades += 1
            self.loss_amounts.append(abs(pnl))
        
        # Store in trade history
        self.trade_history.append({
            'pnl': pnl,
            'is_win': is_win,
            'capital_after': self.current_capital
        })
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        logger.info(f"Trade recorded: PnL=${pnl:.2f}, Capital=${self.current_capital:.2f}, Win Rate: {self.get_win_rate()*100:.1f}%")
    
    def get_win_rate(self) -> float:
        """Get win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_average_win(self) -> float:
        """Get average winning trade from actual history"""
        if len(self.win_amounts) == 0:
            return Config.TAKE_PROFIT_PERCENT * self.initial_capital * Config.RISK_PER_TRADE
        return np.mean(self.win_amounts)
    
    def get_average_loss(self) -> float:
        """Get average losing trade from actual history"""
        if len(self.loss_amounts) == 0:
            return Config.STOP_LOSS_PERCENT * self.initial_capital * Config.RISK_PER_TRADE
        return np.mean(self.loss_amounts)
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(self.loss_amounts) == 0 or sum(self.loss_amounts) == 0:
            return 0.0
        gross_profit = sum(self.win_amounts)
        gross_loss = sum(self.loss_amounts)
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               highest_price: float, direction: str, 
                               atr: Optional[float] = None) -> float:
        """
        Calculate trailing stop to lock in profits
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry (for BUY) or lowest (for SELL)
            direction: 'BUY' or 'SELL'
            atr: Average True Range for dynamic trailing
            
        Returns:
            New trailing stop price
        """
        if atr and atr > 0:
            # ATR-based trailing stop (1.5x ATR from peak)
            trail_distance = atr * 1.5
        else:
            # Fixed 2% trailing stop
            trail_distance = highest_price * 0.02
        
        if direction == 'BUY':
            # For longs, trail below highest price
            trailing_stop = highest_price - trail_distance
            # Never move stop down
            initial_stop = self.calculate_stop_loss(entry_price, direction, atr)
            return max(trailing_stop, initial_stop)
        else:
            # For shorts, trail above lowest price
            trailing_stop = highest_price + trail_distance
            initial_stop = self.calculate_stop_loss(entry_price, direction, atr)
            return min(trailing_stop, initial_stop)
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if self.peak_capital == 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.is_trading_enabled = True
        logger.info("Daily statistics reset")
    
    def get_stats(self) -> dict:
        """Get comprehensive risk management statistics"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_pnl': self.current_capital - self.initial_capital,
            'total_pnl_percent': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': (self.daily_pnl / self.initial_capital) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.get_win_rate() * 100,
            'profit_factor': self.get_profit_factor(),
            'avg_win': self.get_average_win(),
            'avg_loss': self.get_average_loss(),
            'current_drawdown': self.get_current_drawdown(),
            'open_positions': self.open_positions,
            'trading_enabled': self.is_trading_enabled
        }
