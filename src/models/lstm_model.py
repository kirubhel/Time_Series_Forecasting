"""
LSTM Model for Time Series Forecasting
Implements LSTM neural networks for financial time series prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM model will not work.")

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for time series forecasting"""
    
    def __init__(self, lookback: int = 60, units: int = 50, dropout: float = 0.2):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
    
    def prepare_data(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(series.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def fit(self, series: pd.Series, validation_split: float = 0.2, 
            epochs: int = 100, batch_size: int = 32) -> 'LSTMModel':
        """Fit LSTM model to the data"""
        logger.info("Fitting LSTM model...")
        
        X, y = self.prepare_data(series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
        return self
    
    def predict(self, series: pd.Series, steps: int = 30) -> Tuple[pd.Series, np.ndarray]:
        """Make predictions using fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        last_sequence = series.values[-self.lookback:].reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(steps):
            X_pred = current_sequence.reshape(1, self.lookback, 1)
            pred = self.model.predict(X_pred, verbose=0)
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred, axis=0)
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array)
        
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        predictions_series = pd.Series(predictions_unscaled.flatten(), index=future_dates)
        return predictions_series, predictions_array
    
    def evaluate(self, test_series: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_test, y_test = self.prepare_data(test_series)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        actual_direction = np.diff(y_actual.flatten()) > 0
        pred_direction = np.diff(y_pred.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse,
            'MAPE': mape, 'Directional_Accuracy': directional_accuracy
        }
        
        return metrics

def main():
    """Main function to demonstrate LSTM modeling"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.collector import FinancialDataCollector
    from data.preprocessor import FinancialDataPreprocessor
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Please install TensorFlow to use LSTM models.")
        return
    
    collector = FinancialDataCollector()
    raw_data = collector.get_combined_data()
    
    if raw_data.empty:
        print("No data available")
        return
    
    preprocessor = FinancialDataPreprocessor()
    clean_data = preprocessor.clean_data(raw_data)
    
    tsla_data = clean_data[clean_data['Symbol'] == 'TSLA'].set_index('Date')['Close']
    
    train_size = int(len(tsla_data) * 0.8)
    train_data = tsla_data[:train_size]
    test_data = tsla_data[train_size:]
    
    print(f"Training data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    
    lstm_model = LSTMModel(lookback=60, units=50, dropout=0.2)
    lstm_model.fit(train_data, epochs=50, batch_size=32)
    
    metrics = lstm_model.evaluate(test_data)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    future_predictions, _ = lstm_model.predict(tsla_data, steps=30)
    print(f"\nFuture predictions (next 30 days):")
    print(f"Mean prediction: {future_predictions.mean():.2f}")
    print(f"Prediction range: {future_predictions.min():.2f} - {future_predictions.max():.2f}")

if __name__ == "__main__":
    main() 