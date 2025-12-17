"""
High Accuracy Trading Strategy
Ultra-conservative approach focusing on win rate over trade frequency
Only trades when multiple strong confirmations align
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
from utils.indicators import TechnicalIndicators
from loguru import logger


class HighAccuracyStrategy(BaseStrategy):
    """
    High accuracy strategy with strict entry criteria
    Aims for 60%+ win rate by only trading perfect setups
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="High Accuracy", **kwargs)
        self.min_score = 8  # Require 8/10 score to trade
        
    def analyze(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, Dict]:
        """
        Analyze market with strict multi-confirmation system
        Returns signal only when confidence is very high
        """
        try:
            if len(df) < 100:
                return "HOLD", 0.0, {"reason": "Insufficient data"}
            
            # Calculate all indicators using TechnicalIndicators class
            df = TechnicalIndicators.calculate_all(df)
            
            # Get current values
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_hist = df['macd_hist'].iloc[-1]
            prev_hist = df['macd_hist'].iloc[-2]
            current_stoch_k = df['stoch_k'].iloc[-1]
            current_stoch_d = df['stoch_d'].iloc[-1]
            current_adx = df['adx'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            current_williams = df['willr'].iloc[-1]
            current_cci = df['cci'].iloc[-1]
            
            current_ema_20 = df['ema_21'].iloc[-1]  # Use ema_21 instead
            current_ema_50 = df['ema_50'].iloc[-1]
            current_ema_200 = df['sma_20'].iloc[-1]  # Use sma_20 as fallback
            
            current_upper = df['bb_upper'].iloc[-1]
            current_lower = df['bb_lower'].iloc[-1]
            current_middle = df['bb_middle'].iloc[-1]
            
            # Calculate trend strength
            price_array = df['close'].values[-20:]
            trend_strength = abs(np.polyfit(range(len(price_array)), price_array, 1)[0])
            
            # BUY SIGNAL SCORING (10 criteria)
            buy_score = 0
            buy_reasons = []
            
            # 1. RSI oversold but not extreme (35-45)
            if 35 <= current_rsi <= 45:
                buy_score += 1
                buy_reasons.append("RSI in buy zone")
            
            # 2. MACD bullish crossover or positive histogram growing
            if current_hist > 0 and current_hist > prev_hist:
                buy_score += 1
                buy_reasons.append("MACD bullish")
            
            # 3. Price near lower Bollinger Band (bounce play)
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            if bb_position < 0.3:
                buy_score += 1
                buy_reasons.append("Near lower BB")
            
            # 4. Stochastic oversold but turning up
            if current_stoch_k < 30 and current_stoch_k > current_stoch_d:
                buy_score += 1
                buy_reasons.append("Stochastic turning bullish")
            
            # 5. Strong trend (ADX > 25)
            if current_adx > 25:
                buy_score += 1
                buy_reasons.append("Strong trend")
            
            # 6. EMA alignment (20 > 50 or price above 20)
            if current_price > current_ema_20 or current_ema_20 > current_ema_50:
                buy_score += 1
                buy_reasons.append("EMA aligned")
            
            # 7. Williams %R oversold (-80 to -60)
            if -80 <= current_williams <= -60:
                buy_score += 1
                buy_reasons.append("Williams oversold")
            
            # 8. CCI showing bullish momentum
            if -100 <= current_cci <= 0:
                buy_score += 1
                buy_reasons.append("CCI bullish")
            
            # 9. Price action: Recent higher low
            recent_lows = df['low'].values[-5:]
            if recent_lows[-1] > recent_lows[-3]:
                buy_score += 1
                buy_reasons.append("Higher low pattern")
            
            # 10. Volume confirmation (if available) or volatility
            avg_atr = df['atr'].iloc[-10:].mean()
            if current_atr > avg_atr * 0.8:  # Sufficient volatility
                buy_score += 1
                buy_reasons.append("Good volatility")
            
            # SELL SIGNAL SCORING (10 criteria)
            sell_score = 0
            sell_reasons = []
            
            # 1. RSI overbought but not extreme (55-65)
            if 55 <= current_rsi <= 65:
                sell_score += 1
                sell_reasons.append("RSI in sell zone")
            
            # 2. MACD bearish crossover or negative histogram growing
            if current_hist < 0 and current_hist < prev_hist:
                sell_score += 1
                sell_reasons.append("MACD bearish")
            
            # 3. Price near upper Bollinger Band (resistance)
            if bb_position > 0.7:
                sell_score += 1
                sell_reasons.append("Near upper BB")
            
            # 4. Stochastic overbought but turning down
            if current_stoch_k > 70 and current_stoch_k < current_stoch_d:
                sell_score += 1
                sell_reasons.append("Stochastic turning bearish")
            
            # 5. Strong trend (ADX > 25)
            if current_adx > 25:
                sell_score += 1
                sell_reasons.append("Strong trend")
            
            # 6. EMA alignment (20 < 50 or price below 20)
            if current_price < current_ema_20 or current_ema_20 < current_ema_50:
                sell_score += 1
                sell_reasons.append("EMA aligned bearish")
            
            # 7. Williams %R overbought (-40 to -20)
            if -40 <= current_williams <= -20:
                sell_score += 1
                sell_reasons.append("Williams overbought")
            
            # 8. CCI showing bearish momentum
            if 0 <= current_cci <= 100:
                sell_score += 1
                sell_reasons.append("CCI bearish")
            
            # 9. Price action: Recent lower high
            recent_highs = df['high'].values[-5:]
            if recent_highs[-1] < recent_highs[-3]:
                sell_score += 1
                sell_reasons.append("Lower high pattern")
            
            # 10. Volume confirmation (if available) or volatility
            if current_atr > avg_atr * 0.8:
                sell_score += 1
                sell_reasons.append("Good volatility")
            
            # DECISION LOGIC - Require 8+ score
            confidence = 0.0
            signal = "HOLD"
            metadata = {}
            
            if buy_score >= self.min_score and buy_score > sell_score:
                signal = "BUY"
                confidence = min(0.95, 0.6 + (buy_score / 20))  # 60-95% confidence
                metadata = {
                    "reason": f"Strong BUY signal ({buy_score}/10)",
                    "score": buy_score,
                    "confirmations": buy_reasons,
                    "rsi": round(current_rsi, 2),
                    "adx": round(current_adx, 2),
                    "macd_hist": round(current_hist, 4),
                    "bb_position": round(bb_position, 2)
                }
                logger.info(f"HIGH ACCURACY BUY: Score {buy_score}/10 - {', '.join(buy_reasons[:3])}")
                
            elif sell_score >= self.min_score and sell_score > buy_score:
                signal = "SELL"
                confidence = min(0.95, 0.6 + (sell_score / 20))
                metadata = {
                    "reason": f"Strong SELL signal ({sell_score}/10)",
                    "score": sell_score,
                    "confirmations": sell_reasons,
                    "rsi": round(current_rsi, 2),
                    "adx": round(current_adx, 2),
                    "macd_hist": round(current_hist, 4),
                    "bb_position": round(bb_position, 2)
                }
                logger.info(f"HIGH ACCURACY SELL: Score {sell_score}/10 - {', '.join(sell_reasons[:3])}")
                
            else:
                # No clear signal
                metadata = {
                    "reason": "No strong confirmation",
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "required": self.min_score,
                    "rsi": round(current_rsi, 2),
                    "adx": round(current_adx, 2)
                }
            
            return signal, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in high accuracy analysis: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def train(self, historical_data: pd.DataFrame):
        """No training needed for this rule-based strategy"""
        logger.info(f"{self.name} strategy initialized - Rule-based system ready")
        logger.info(f"Minimum score required: {self.min_score}/10 confirmations")
        return True
