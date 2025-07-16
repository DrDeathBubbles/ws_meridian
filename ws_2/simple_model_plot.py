#!/usr/bin/env python3
"""
Simple Model Prediction Plot
===========================
Creates a minimal plot comparing actual data vs model predictions.
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

def get_model_predictions(model, actual_data):
    """Extract real predictions from the model"""
    print("\nExtracting model predictions...")
    
    # Print available attributes to help debug
    print("Available model attributes:")
    model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
    print(", ".join(model_attrs[:20]) + ("..." if len(model_attrs) > 20 else ""))
    
    # Try to access the model's analyze method
    if hasattr(model, 'analyze') and callable(getattr(model, 'analyze')):
        try:
            print("Trying to use model.analyze() method...")
            analyzer = model.analyze()
            print(f"Got analyzer with type {type(analyzer)}")
            
            # If analyzer has a predict method, use it
            if hasattr(analyzer, 'predict') and callable(getattr(analyzer, 'predict')):
                print("Found predict method in analyzer")
                predictions = analyzer.predict()
                print(f"Got predictions with type {type(predictions)} and shape {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                return predictions
        except Exception as e:
            print(f"Error using analyze method: {e}")
    
    # Check if model has posterior_samples
    if hasattr(model, 'posterior_samples'):
        posterior = model.posterior_samples
        print(f"Found posterior_samples with type {type(posterior)}")
        
        # If it's a dict, print the keys
        if isinstance(posterior, dict):
            print(f"Posterior samples keys: {list(posterior.keys())}")
            
            # Look for common parameter names that might contain predictions
            for param_name in ['mu', 'y_pred', 'prediction', 'kpi_pred', 'y_hat']:
                if param_name in posterior:
                    print(f"Found '{param_name}' in posterior samples with shape {posterior[param_name].shape}")
                    # Calculate mean across samples (first dimension)
                    predictions = np.mean(posterior[param_name], axis=0)
                    print(f"Extracted predictions with shape {predictions.shape}")
                    return predictions
    
    # Try to access the model's predict method directly
    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        try:
            print("Trying model.predict() method...")
            predictions = model.predict()
            print(f"Got predictions with type {type(predictions)} and shape {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
            return predictions
        except Exception as e:
            print(f"Error calling predict method: {e}")
    
    # If we have a model_spec, try to use it to generate predictions
    if hasattr(model, 'model_spec') and hasattr(model, 'input_data'):
        print("Attempting to generate predictions using model_spec and input_data...")
        try:
            # If we have a total_outcome attribute, it might contain the fitted values
            if hasattr(model, 'total_outcome'):
                print("Found total_outcome attribute, checking if it contains predictions...")
                total_outcome = model.total_outcome
                if hasattr(total_outcome, 'numpy') and callable(getattr(total_outcome, 'numpy')):
                    predictions = total_outcome.numpy()
                    print(f"Extracted predictions from total_outcome with type {type(predictions)}")
                    
                    # If predictions is a scalar, create an array of that value
                    if np.isscalar(predictions):
                        print(f"Predictions is a scalar value: {predictions}")
                        print("Creating an array of this value with the same length as actual_data")
                        predictions = np.full_like(actual_data, predictions)
                    elif len(predictions.shape) > 1:
                        predictions = np.mean(predictions, axis=0)  # Average across first dimension if needed
                    
                    return predictions
        except Exception as e:
            print(f"Error generating predictions: {e}")
    
    # If we have a baseline_summary_metrics method, try to use it
    if hasattr(model, 'baseline_summary_metrics') and callable(model.baseline_summary_metrics):
        try:
            print("Trying baseline_summary_metrics method...")
            metrics = model.baseline_summary_metrics()
            print(f"Got metrics with type {type(metrics)}")
            # This might contain the baseline predictions which we could use
        except Exception as e:
            print(f"Error calling baseline_summary_metrics: {e}")
    
    # If we have an analyze method, try to use it
    if hasattr(model, 'analyze') and callable(model.analyze):
        try:
            print("Trying analyze method...")
            analysis = model.analyze()
            print(f"Got analysis with type {type(analysis)}")
            # This might contain predictions or fitted values
        except Exception as e:
            print(f"Error calling analyze: {e}")
    
    # If all else fails, we need to inform the user
    print("\nWARNING: Could not extract real predictions from the model.")
    print("Please provide more information about the model structure or how to access predictions.")
    
    # Return None to indicate we couldn't get real predictions
    return None

def main():
    print("Loading model...")
    
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Creating plot...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Extract data from model
    try:
        # Try to get actual data
        if hasattr(model, 'input_data') and hasattr(model.input_data, 'kpi'):
            actual_data = model.input_data.kpi
            print(f"Found actual data with shape: {actual_data.shape}")
            print(f"Data type: {type(actual_data)}")
            
            # Handle DataArray objects
            if XARRAY_AVAILABLE and isinstance(actual_data, xr.DataArray):
                print("Converting DataArray to numpy array")
                actual_data = actual_data.values
                print(f"After conversion, shape: {actual_data.shape}")
            
            # Ensure data is 1D by flattening
            actual_data = actual_data.reshape(-1)
            print(f"After reshaping to 1D, shape: {actual_data.shape}")
        else:
            # Create dummy data if we can't find actual data
            print("Could not find actual data, using dummy data")
            actual_data = np.random.rand(20)
        
        # Get model predictions
        predictions = get_model_predictions(model, actual_data)
        
        # If we couldn't get real predictions, just plot the actual data
        if predictions is None:
            print("\nPlotting only the actual data since we couldn't extract predictions.")
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create x values (simple indices)
            x = np.arange(len(actual_data))
            
            # Plot actual data - use only every 20th point for markers to avoid clutter
            plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
            plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
            
            # Format plot
            plt.title('Actual Data', fontsize=14, fontweight='bold')
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Save plot
            output_path = os.path.join(OUTPUT_DIR, 'actual_data.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {output_path}")
            return
        
        # Ensure predictions are in the right format
        if np.isscalar(predictions):
            print(f"Predictions is a scalar value: {predictions}")
            print("Creating an array of this value with the same length as actual_data")
            predictions = np.full_like(actual_data, predictions)
        elif hasattr(predictions, 'shape') and len(predictions.shape) > 1:
            predictions = predictions.reshape(-1)
        
        # Make sure predictions match actual data length
        if len(predictions) != len(actual_data):
            print(f"Warning: predictions length {len(predictions)} doesn't match actual data length {len(actual_data)}")
            # Use the shorter length
            min_len = min(len(predictions), len(actual_data))
            actual_data = actual_data[:min_len]
            predictions = predictions[:min_len]
        
        # Create x values (simple indices)
        x = np.arange(len(actual_data))
        
        # Plot actual data - use only every 10th point for markers to avoid clutter
        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
        
        # Plot predictions
        plt.plot(x, predictions, '-', color='red', label='Model Predictions')
        
        # Add confidence interval (simple approximation)
        std_dev = np.std(actual_data - predictions)
        lower = predictions - 1.96 * std_dev  # 95% CI
        upper = predictions + 1.96 * std_dev  # 95% CI
        plt.fill_between(x, lower, upper, color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Calculate metrics
        mse = np.mean((actual_data - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_data - predictions))
        
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
        plt.title('Model Predictions vs Actual Data', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'model_predictions_vs_actual.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        print(f"Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()