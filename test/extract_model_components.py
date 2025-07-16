#!/usr/bin/env python3
"""
Extract Model Components
=======================
Extracts and analyzes the components of the Meridian model to find time-varying predictions.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Set paths
MODEL_PATH = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model/improved_model.pkl'
OUTPUT_DIR = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'

def main():
    print("Loading model...")
    
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Analyzing model structure...")
    
    # Get actual data
    if hasattr(model, 'input_data') and hasattr(model.input_data, 'kpi'):
        actual_data = model.input_data.kpi
        print(f"Found actual data with shape: {actual_data.shape}")
        
        # Convert to numpy array if needed
        if hasattr(actual_data, 'values'):
            actual_data = actual_data.values
        
        # Reshape to 1D
        actual_data = actual_data.reshape(-1)
        print(f"Actual data shape after reshaping: {actual_data.shape}")
    else:
        print("Could not find actual data")
        return
    
    # Try to extract model components
    components = {}
    
    # Check for media effect components
    if hasattr(model, 'media_effect'):
        print("Found media_effect attribute")
        media_effect = model.media_effect
        if hasattr(media_effect, 'numpy'):
            media_effect = media_effect.numpy()
        components['media_effect'] = media_effect
    
    # Check for baseline components
    if hasattr(model, 'baseline'):
        print("Found baseline attribute")
        baseline = model.baseline
        if hasattr(baseline, 'numpy'):
            baseline = baseline.numpy()
        components['baseline'] = baseline
    
    # Check for trend components
    if hasattr(model, 'trend'):
        print("Found trend attribute")
        trend = model.trend
        if hasattr(trend, 'numpy'):
            trend = trend.numpy()
        components['trend'] = trend
    
    # Check for seasonality components
    if hasattr(model, 'seasonality'):
        print("Found seasonality attribute")
        seasonality = model.seasonality
        if hasattr(seasonality, 'numpy'):
            seasonality = seasonality.numpy()
        components['seasonality'] = seasonality
    
    # Check for posterior samples
    if hasattr(model, 'posterior_samples'):
        print("Found posterior_samples")
        posterior = model.posterior_samples
        if isinstance(posterior, dict):
            print(f"Posterior samples keys: {list(posterior.keys())}")
            
            # Try to extract components from posterior samples
            for key in posterior.keys():
                print(f"  - {key}: {posterior[key].shape if hasattr(posterior[key], 'shape') else 'N/A'}")
                components[f'posterior_{key}'] = posterior[key]
    
    # Check for linear predictor components
    for attr_name in dir(model):
        if 'linear_predictor' in attr_name and not attr_name.startswith('_'):
            print(f"Found {attr_name}")
            component = getattr(model, attr_name)
            if hasattr(component, 'numpy'):
                component = component.numpy()
            components[attr_name] = component
    
    # Print found components
    print("\nFound components:")
    for name, component in components.items():
        if hasattr(component, 'shape'):
            print(f"  - {name}: shape {component.shape}")
        else:
            print(f"  - {name}: {type(component)}")
    
    # Try to reconstruct time-varying predictions
    print("\nAttempting to reconstruct time-varying predictions...")
    
    # Try different approaches to get time-varying predictions
    predictions = None
    
    # Approach 1: Use linear_predictor if available
    if 'linear_predictor' in components:
        print("Using linear_predictor")
        predictions = components['linear_predictor']
        if len(predictions.shape) > 1:
            # Average across samples or geos if needed
            predictions = np.mean(predictions, axis=tuple(range(len(predictions.shape) - 1)))
    
    # Approach 2: Use media_effect + baseline if available
    elif 'media_effect' in components and 'baseline' in components:
        print("Using media_effect + baseline")
        media_effect = components['media_effect']
        baseline = components['baseline']
        
        # Reshape if needed
        if len(media_effect.shape) > 1:
            media_effect = np.mean(media_effect, axis=tuple(range(len(media_effect.shape) - 1)))
        if len(baseline.shape) > 1:
            baseline = np.mean(baseline, axis=tuple(range(len(baseline.shape) - 1)))
            
        # Add components
        if len(media_effect) == len(baseline):
            predictions = media_effect + baseline
        else:
            print(f"Shape mismatch: media_effect {media_effect.shape}, baseline {baseline.shape}")
    
    # Approach 3: Use posterior mu if available
    elif 'posterior_mu' in components:
        print("Using posterior_mu")
        posterior_mu = components['posterior_mu']
        if len(posterior_mu.shape) > 1:
            # Average across samples
            predictions = np.mean(posterior_mu, axis=0)
    
    # If we found predictions, create a plot
    if predictions is not None and len(predictions) > 1:
        print(f"Found time-varying predictions with shape {predictions.shape}")
        
        # Make sure predictions match actual data length
        if len(predictions) != len(actual_data):
            print(f"Warning: predictions length {len(predictions)} doesn't match actual data length {len(actual_data)}")
            # Use the shorter length
            min_len = min(len(predictions), len(actual_data))
            actual_data = actual_data[:min_len]
            predictions = predictions[:min_len]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create x values (simple indices)
        x = np.arange(len(actual_data))
        
        # Plot actual data
        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
        
        # Plot predictions
        plt.plot(x, predictions, '-', color='red', label='Model Predictions')
        
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
        output_path = os.path.join(OUTPUT_DIR, 'model_components_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        print(f"Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    else:
        print("Could not find time-varying predictions")

if __name__ == "__main__":
    main()