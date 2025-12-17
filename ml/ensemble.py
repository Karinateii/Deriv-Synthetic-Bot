"""
Ensemble Learning for Price Prediction
Combines multiple ML models for better accuracy
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from loguru import logger
import joblib
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn/XGBoost/LightGBM not available. Ensemble learning disabled.")


class EnsemblePredictor:
    """Ensemble of multiple ML models for prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
        if not SKLEARN_AVAILABLE:
            logger.warning("ML libraries not available. Ensemble predictor disabled.")
            return
            
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from price data
        
        Args:
            df: DataFrame with OHLC data and indicators
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['price'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # Moving averages
        for period in [9, 21, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # Technical indicators (if available)
        if 'rsi' in df.columns:
            features['rsi'] = df['rsi']
            
        if 'macd' in df.columns:
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_diff'] = df['macd'] - df['macd_signal']
            
        if 'bb_upper' in df.columns:
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
        if 'atr' in df.columns:
            features['atr'] = df['atr']
            features['atr_ratio'] = df['atr'] / df['close']
            
        if 'adx' in df.columns:
            features['adx'] = df['adx']
            
        if 'stoch_k' in df.columns:
            features['stoch_k'] = df['stoch_k']
            features['stoch_d'] = df['stoch_d']
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_ma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
        
        return features.dropna()
    
    def prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create labels for classification
        
        Returns:
            Array of labels: 0=DOWN, 1=NEUTRAL, 2=UP
        """
        future_returns = df['close'].pct_change().shift(-1)
        
        labels = np.zeros(len(future_returns))
        labels[future_returns > 0.0005] = 2  # UP
        labels[future_returns < -0.0005] = 0  # DOWN
        labels[(future_returns >= -0.0005) & (future_returns <= 0.0005)] = 1  # NEUTRAL
        
        # Remove NaN values
        return labels[:-1]  # Remove last element (has no future return)
    
    def train(self, df: pd.DataFrame):
        """
        Train all ensemble models
        
        Args:
            df: DataFrame with OHLC data and indicators
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train: ML libraries not available")
            return
            
        try:
            logger.info("Creating features for ensemble training...")
            features = self.create_features(df)
            labels = self.prepare_labels(df)
            
            # Make sure features and labels have same length
            min_length = min(len(features), len(labels))
            features = features.iloc[:min_length]
            labels = labels[:min_length]
            
            if len(features) < 100:
                logger.warning("Insufficient data for training ensemble")
                return
            
            # Scale features
            X = self.scaler.fit_transform(features)
            y = labels
            
            # Train each model
            logger.info("Training ensemble models...")
            for name, model in self.models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X, y)
                    score = model.score(X, y)
                    logger.success(f"{name} trained. Accuracy: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            self.is_trained = True
            logger.success("Ensemble training completed")
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {e}")
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict using ensemble voting
        
        Args:
            df: Recent DataFrame with OHLC data and indicators
            
        Returns:
            (prediction, confidence) - Direction and confidence
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return 'NEUTRAL', 0.0
            
        try:
            # Create features
            features = self.create_features(df)
            
            if features.empty:
                return 'NEUTRAL', 0.0
            
            # Take last row
            X = features.iloc[-1:].values
            X = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = []
            probabilities = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]
                    predictions.append(pred)
                    probabilities.append(prob)
                except Exception as e:
                    logger.warning(f"Prediction error in {name}: {e}")
            
            if not predictions:
                return 'NEUTRAL', 0.0
            
            # Voting
            predictions_array = np.array(predictions)
            votes = np.bincount(predictions_array.astype(int))
            
            # Get majority vote
            predicted_class = np.argmax(votes)
            confidence = votes[predicted_class] / len(predictions)
            
            # Average probabilities for confidence
            avg_probs = np.mean(probabilities, axis=0)
            confidence = float(avg_probs[predicted_class])
            
            directions = ['DOWN', 'NEUTRAL', 'UP']
            direction = directions[predicted_class]
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 'NEUTRAL', 0.0
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from Random Forest"""
        if not self.is_trained or 'random_forest' not in self.models:
            return {}
            
        try:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            return dict(zip(range(len(importances)), importances))
        except:
            return {}
    
    def save(self, filepath: Path):
        """Save models to disk"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            for name, model in self.models.items():
                joblib.dump(model, filepath / f'{name}.pkl')
            joblib.dump(self.scaler, filepath / 'ensemble_scaler.pkl')
            logger.info(f"Ensemble models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
    
    def load(self, filepath: Path):
        """Load models from disk"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            for name in self.models.keys():
                model_path = filepath / f'{name}.pkl'
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
            
            scaler_path = filepath / 'ensemble_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                
            self.is_trained = True
            logger.success(f"Ensemble models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
