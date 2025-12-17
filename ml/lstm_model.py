"""
LSTM Neural Network for Price Prediction
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import joblib
from pathlib import Path
from loguru import logger

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. ML predictions will be disabled.")

from config import Config


class LSTMPredictor:
    """LSTM model for price prediction"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model: Optional[Sequential] = None
        self.scaler = None
        self.is_trained = False
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. LSTM predictor disabled.")
            
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(3, activation='softmax')  # UP, DOWN, NEUTRAL
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("LSTM model built successfully")
        
    def prepare_data(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training/prediction
        
        Args:
            prices: Array of price data
            
        Returns:
            (X, y) - Features and labels
        """
        from sklearn.preprocessing import StandardScaler
        
        # Normalize data
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
        else:
            scaled_data = self.scaler.transform(prices.reshape(-1, 1))
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            
            # Calculate price change
            future_price = prices[i]
            current_price = prices[i-1]
            price_change = (future_price - current_price) / current_price
            
            # Classify: 0=DOWN, 1=NEUTRAL, 2=UP
            if price_change > 0.0005:  # 0.05% threshold
                y.append([0, 0, 1])  # UP
            elif price_change < -0.0005:
                y.append([1, 0, 0])  # DOWN
            else:
                y.append([0, 1, 0])  # NEUTRAL
        
        return np.array(X), np.array(y)
    
    def train(self, prices: np.ndarray, validation_split: float = 0.2):
        """
        Train the LSTM model
        
        Args:
            prices: Historical price data
            validation_split: Fraction of data for validation
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("Cannot train: TensorFlow not available")
            return
            
        if len(prices) < Config.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for training. Need at least {Config.MIN_TRAINING_SAMPLES} samples")
            return
        
        try:
            logger.info("Preparing training data...")
            X, y = self.prepare_data(prices)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build model if not exists
            if self.model is None:
                self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            logger.info("Training LSTM model...")
            history = self.model.fit(
                X, y,
                epochs=Config.LSTM_EPOCHS,
                batch_size=Config.LSTM_BATCH_SIZE,
                validation_split=validation_split,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            self.is_trained = True
            logger.success(f"Model trained successfully. Final accuracy: {history.history['accuracy'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def predict(self, recent_prices: np.ndarray) -> Tuple[str, float]:
        """
        Predict next price direction
        
        Args:
            recent_prices: Recent price data (should be >= sequence_length)
            
        Returns:
            (prediction, confidence) - Direction ('UP', 'DOWN', 'NEUTRAL') and confidence (0-1)
        """
        if not TENSORFLOW_AVAILABLE or not self.is_trained or self.model is None:
            return 'NEUTRAL', 0.0
            
        try:
            # Take last sequence_length prices
            if len(recent_prices) < self.sequence_length:
                return 'NEUTRAL', 0.0
                
            sequence = recent_prices[-self.sequence_length:]
            
            # Scale data
            scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))
            
            # Reshape for prediction
            X = scaled_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict
            prediction = self.model.predict(X, verbose=0)[0]
            
            # Get direction and confidence
            class_idx = np.argmax(prediction)
            confidence = float(prediction[class_idx])
            
            directions = ['DOWN', 'NEUTRAL', 'UP']
            direction = directions[class_idx]
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 'NEUTRAL', 0.0
    
    def save(self, filepath: Path):
        """Save model to disk"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return
            
        try:
            self.model.save(filepath / 'lstm_model.h5')
            joblib.dump(self.scaler, filepath / 'scaler.pkl')
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self, filepath: Path):
        """Load model from disk"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        try:
            model_path = filepath / 'lstm_model.h5'
            scaler_path = filepath / 'scaler.pkl'
            
            if model_path.exists() and scaler_path.exists():
                self.model = keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                logger.success(f"Model loaded from {filepath}")
            else:
                logger.warning(f"Model files not found in {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
