#!/usr/bin/env python3
"""
Plot mu_t vs Actual Data
=======================
Extracts the mu_t variable from the model's posterior and plots it against the actual data.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Get mu_t from inference data
    if hasattr(model, 'inference_data') and hasattr(model.inference_data, 'posterior'):
        posterior = model.inference_data.posterior
        if hasattr(posterior, 'mu_t'):
            print("Found mu_t in posterior")
            mu_t = posterior.mu_t.values
            print(f"mu_t shape: {mu_t.shape}")
            
            # Average across chains and samples
            prediction = mu_t.mean(axis=(0, 1))
            print(f"Prediction shape after averaging: {prediction.shape}")
            
            # Make sure prediction matches actual data length
            if len(prediction) != len(actual_data):
                print(f"Warning: prediction length {len(prediction)} doesn't match actual data length {len(actual_data)}")
                # Use the shorter length
                min_len = min(len(prediction), len(actual_data))
                actual_data = actual_data[:min_len]
                prediction = prediction[:min_len]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Create x values (simple indices)
            x = np.arange(len(actual_data))
            
            # Plot actual data
            plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
            plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
            
            # Plot prediction
            plt.plot(x, prediction, '-', color='red', label='Model Prediction (mu_t)')
            
            # Calculate metrics
            mse = np.mean((actual_data - prediction) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_data - prediction))
            
            # Add metrics to plot
            metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
            plt.annotate(
                metrics_text,
                xy=(0.02, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10
            )
            
            # Format plot
            plt.title('Model Predictions (mu_t) vs Actual Data', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='upper right')
            
            # Save plot
            output_path = os.path.join(OUTPUT_DIR, 'mu_t_vs_actual.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {output_path}")
            print(f"Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            # Also create a plot with confidence intervals
            plt.figure(figsize=(12, 8))
            
            # Plot actual data
            plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
            plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
            
            # Plot prediction
            plt.plot(x, prediction, '-', color='red', label='Model Prediction (mu_t)')
            
            # Calculate 95% confidence intervals
            lower = np.percentile(mu_t, 2.5, axis=(0, 1))
            upper = np.percentile(mu_t, 97.5, axis=(0, 1))
            
            # Plot confidence intervals
            plt.fill_between(x, lower, upper, color='red', alpha=0.2, label='95% Confidence Interval')
            
            # Add metrics to plot
            plt.annotate(
                metrics_text,
                xy=(0.02, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10
            )
            
            # Format plot
            plt.title('Model Predictions with Confidence Intervals vs Actual Data', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='upper right')
            
            # Save plot
            output_path = os.path.join(OUTPUT_DIR, 'mu_t_with_ci_vs_actual.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plot with confidence intervals saved to: {output_path}")
        else:
            print("Could not find mu_t in posterior")
    else:
        print("Could not find inference_data or posterior")

if __name__ == "__main__":
    main()