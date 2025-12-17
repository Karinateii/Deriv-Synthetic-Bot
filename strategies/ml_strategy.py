"""
ML-Enhanced Trading Strategy
Combines machine learning predictions with technical analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger
from strategies.base_strategy import BaseStrategy
from ml.lstm_model import LSTMPredictor
from ml.ensemble import EnsemblePredictor
from config import Config


class MLStrategy(BaseStrategy):
    """Machine Learning Strategy with Ensemble and LSTM"""
    
    def __init__(self):
        super().__init__("ML_Ensemble")
        self.lstm = LSTMPredictor(sequence_length=Config.LSTM_SEQUENCE_LENGTH)
        self.ensemble = EnsemblePredictor()
        self.lstm_weight = 0.4
        self.ensemble_weight = 0.6
        
    def train(self, df: pd.DataFrame):
        """Train both ML models"""
        try:
            logger.info("Training ML models...")
            
            # Train LSTM on price data
            if len(df) >= Config.MIN_TRAINING_SAMPLES:
                prices = df['close'].values
                self.lstm.train(prices)
            
            # Train ensemble on features
            self.ensemble.train(df)
            
            logger.success("ML models training completed")
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
    
    def analyze(self, df: pd.DataFrame, current_price: float) -> Tuple[str, float, dict]:
        """
        Analyze using ML predictions and technical confirmation
        
        Returns:
            (signal, confidence, metadata)
        """
        if df.empty or len(df) < 50:
            return 'NEUTRAL', 0.0, {'reason': 'Insufficient data'}
        
        metadata = {}
        
        # Get LSTM prediction
        lstm_signal = 'NEUTRAL'
        lstm_confidence = 0.0
        
        if self.lstm.is_trained:
            prices = df['close'].values
            lstm_direction, lstm_conf = self.lstm.predict(prices)
            lstm_confidence = lstm_conf
            
            if lstm_direction == 'UP':
                lstm_signal = 'BUY'
            elif lstm_direction == 'DOWN':
                lstm_signal = 'SELL'
            
            metadata['lstm_signal'] = lstm_signal
            metadata['lstm_confidence'] = lstm_confidence
        
        # Get Ensemble prediction
        ensemble_signal = 'NEUTRAL'
        ensemble_confidence = 0.0
        
        if self.ensemble.is_trained:
            ensemble_direction, ensemble_conf = self.ensemble.predict(df)
            ensemble_confidence = ensemble_conf
            
            if ensemble_direction == 'UP':
                ensemble_signal = 'BUY'
            elif ensemble_direction == 'DOWN':
                ensemble_signal = 'SELL'
            
            metadata['ensemble_signal'] = ensemble_signal
            metadata['ensemble_confidence'] = ensemble_confidence
        
        # Combine predictions
        if lstm_signal == ensemble_signal and lstm_signal != 'NEUTRAL':
            # Both agree
            final_signal = lstm_signal
            final_confidence = (lstm_confidence * self.lstm_weight + 
                              ensemble_confidence * self.ensemble_weight)
            metadata['agreement'] = 'full'
        elif lstm_signal == 'NEUTRAL' or ensemble_signal == 'NEUTRAL':
            # One neutral
            final_signal = lstm_signal if ensemble_signal == 'NEUTRAL' else ensemble_signal
            final_confidence = lstm_confidence if ensemble_signal == 'NEUTRAL' else ensemble_confidence
            final_confidence *= 0.7  # Reduce confidence when only one model signals
            metadata['agreement'] = 'partial'
        else:
            # Disagree
            final_signal = 'NEUTRAL'
            final_confidence = 0.0
            metadata['agreement'] = 'none'
        
        # Technical confirmation
        technical_score = self._technical_confirmation(df)
        metadata['technical_score'] = technical_score
        
        # Adjust confidence based on technical indicators
        if final_signal == 'BUY' and technical_score > 0.5:
            final_confidence *= 1.2
        elif final_signal == 'SELL' and technical_score > 0.5:
            final_confidence *= 1.2
        else:
            final_confidence *= 0.8
        
        # Clamp confidence
        final_confidence = min(1.0, final_confidence)
        
        # Check minimum confidence threshold
        if final_confidence < Config.MIN_CONFIDENCE_SCORE:
            final_signal = 'NEUTRAL'
            metadata['reason'] = 'Below confidence threshold'
        
        return final_signal, final_confidence, metadata
    
    def _technical_confirmation(self, df: pd.DataFrame) -> float:
        """
        Score technical indicators for confirmation
        
        Returns:
            Score 0.0 to 1.0
        """
        if df.empty:
            return 0.0
        
        latest = df.iloc[-1]
        score = 0
        max_score = 0
        
        # RSI confirmation
        if 'rsi' in latest:
            max_score += 1
            if 30 <= latest['rsi'] <= 70:
                score += 1
            elif latest['rsi'] < 20 or latest['rsi'] > 80:
                score += 0.5
        
        # MACD confirmation
        if 'macd' in latest and 'macd_signal' in latest:
            max_score += 1
            if latest['macd'] > latest['macd_signal']:
                score += 1
        
        # Trend confirmation
        if 'ema_9' in latest and 'ema_21' in latest:
            max_score += 1
            if latest['ema_9'] > latest['ema_21']:
                score += 1
        
        # ADX strength
        if 'adx' in latest:
            max_score += 1
            if latest['adx'] > 25:
                score += 1
        
        return score / max_score if max_score > 0 else 0.5
