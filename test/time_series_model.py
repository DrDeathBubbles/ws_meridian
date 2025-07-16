#!/usr/bin/env python3
"""
Time Series Model
================
Creates a time series model that captures patterns in the data and compares it to the actual data.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Try to import xarray for DataArray handling
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Set paths
MODEL_PATH = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model/improved_model.pkl'
OUTPUT_DIR = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'

def extract_tensor_value(tensor):
    """Extract value from TensorFlow tensor or xarray DataArray"""
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    elif XARRAY_AVAILABLE and isinstance(tensor, xr.DataArray):
        return tensor.values
    return tensor

def main():
    print("Loading model...")
    
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Extracting data...")
    
    # Get actual data
    if hasattr(model, 'input_data') and hasattr(model.input_data, 'kpi'):
        actual_data = model.input_data.kpi
        print(f"Found actual data with shape: {actual_data.shape}")
        actual_data = extract_tensor_value(actual_data)
        actual_data = actual_data.reshape(-1)
        print(f"Actual data shape after reshaping: {actual_data.shape}")
    else:
        print("Could not find actual data")
        return
    
    # Get mu_t from inference data for comparison
    if hasattr(model, 'inference_data') and hasattr(model.inference_data, 'posterior'):
        posterior = model.inference_data.posterior
        if hasattr(posterior, 'mu_t'):
            print("Found mu_t in posterior")
            mu_t = posterior.mu_t.values
            print(f"mu_t shape: {mu_t.shape}")
            
            # Average across chains and samples
            meridian_prediction = mu_t.mean(axis=(0, 1))
            print(f"Meridian prediction shape after averaging: {meridian_prediction.shape}")
        else:
            meridian_prediction = None
    else:
        meridian_prediction = None
    
    print("Creating time series model...")
    
    # Try to decompose the time series
    try:
        # Decompose the time series into trend, seasonal, and residual components
        print("Decomposing time series...")
        decomposition = seasonal_decompose(actual_data, model='additive', period=7)
        
        # Get components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Fill NaN values in trend
        trend = np.nan_to_num(trend, nan=np.nanmean(trend))
        
        # Create a simple prediction by combining trend and seasonal components
        simple_prediction = trend + seasonal
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create x values (simple indices)
        x = np.arange(len(actual_data))
        
        # Plot actual data
        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
        
        # Plot time series prediction
        plt.plot(x, simple_prediction, '-', color='green', label='Time Series Prediction')
        
        # Plot Meridian prediction if available
        if meridian_prediction is not None:
            plt.plot(x, meridian_prediction, '-', color='red', label='Meridian Prediction')
        
        # Calculate metrics for time series prediction
        mse = np.mean((actual_data - simple_prediction) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_data - simple_prediction))
        
        # Add metrics to plot
        metrics_text = f"Time Series Model Metrics:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
        plt.annotate(
            metrics_text,
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10
        )
        
        # Format plot
        plt.title('Time Series Prediction vs Actual Data', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'time_series_vs_actual.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        print(f"Time Series Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        
        # Create a plot showing the components
        plt.figure(figsize=(12, 12))
        
        # Plot actual data
        plt.subplot(4, 1, 1)
        plt.plot(x, actual_data, '-', color='blue')
        plt.title('Actual Data')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot trend
        plt.subplot(4, 1, 2)
        plt.plot(x, trend, '-', color='red')
        plt.title('Trend Component')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot seasonal
        plt.subplot(4, 1, 3)
        plt.plot(x, seasonal, '-', color='green')
        plt.title('Seasonal Component')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Plot residual
        plt.subplot(4, 1, 4)
        plt.plot(x, residual, '-', color='purple')
        plt.title('Residual Component')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'time_series_components.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Components plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating time series model: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a simpler approach - moving average
        print("Trying moving average approach...")
        
        # Create a simple moving average
        window_size = 7
        weights = np.ones(window_size) / window_size
        ma_prediction = np.convolve(actual_data, weights, mode='same')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create x values (simple indices)
        x = np.arange(len(actual_data))
        
        # Plot actual data
        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
        
        # Plot moving average prediction
        plt.plot(x, ma_prediction, '-', color='green', label=f'Moving Average (window={window_size})')
        
        # Plot Meridian prediction if available
        if meridian_prediction is not None:
            plt.plot(x, meridian_prediction, '-', color='red', label='Meridian Prediction')
        
        # Calculate metrics for moving average prediction
        mse = np.mean((actual_data - ma_prediction) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_data - ma_prediction))
        
        # Add metrics to plot
        metrics_text = f"Moving Average Metrics:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
        plt.annotate(
            metrics_text,
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10
        )
        
        # Format plot
        plt.title('Moving Average vs Actual Data', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'moving_average_vs_actual.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Moving average plot saved to: {output_path}")
        print(f"Moving Average Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

if __name__ == "__main__":
    main()