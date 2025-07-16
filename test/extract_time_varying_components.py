#!/usr/bin/env python3
"""
Extract Time-Varying Components
==============================
Extracts time-varying components from the Meridian model to create a better prediction plot.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif XARRAY_AVAILABLE and isinstance(tensor, xr.DataArray):
        return tensor.values
    return tensor

def main():
    print("Loading model...")
    
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Extracting time-varying components...")
    
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
    
    # Create a dictionary to store all time-varying components
    components = {}
    
    # Try to extract time-varying components using TensorFlow graph execution
    if hasattr(model, 'model_spec') and hasattr(model, 'posterior_samples'):
        print("Found model_spec and posterior_samples")
        
        # Get posterior samples
        posterior = model.posterior_samples
        
        # Check if we have media effect parameters
        if 'beta_media' in posterior:
            print("Found beta_media in posterior samples")
            beta_media = posterior['beta_media']
            print(f"beta_media shape: {beta_media.shape}")
            
            # Get media data
            if hasattr(model, 'media_tensors') and hasattr(model.media_tensors, 'media'):
                media = extract_tensor_value(model.media_tensors.media)
                print(f"media shape: {media.shape}")
                
                # Calculate media effect for each sample
                media_effects = []
                for i in range(min(10, beta_media.shape[0])):  # Use up to 10 samples
                    beta = beta_media[i]
                    # Reshape beta if needed
                    if len(beta.shape) > 1:
                        beta = np.mean(beta, axis=0)
                    
                    # Calculate media effect (simplified)
                    if len(media.shape) == 3:  # [geo, time, channel]
                        effect = np.sum(media * beta.reshape(1, 1, -1), axis=2)
                    else:
                        print(f"Unexpected media shape: {media.shape}")
                        continue
                    
                    media_effects.append(effect)
                
                if media_effects:
                    # Average across samples
                    media_effect = np.mean(media_effects, axis=0)
                    
                    # If we have multiple geos, average across them
                    if len(media_effect.shape) > 1:
                        media_effect = np.mean(media_effect, axis=0)
                    
                    components['media_effect'] = media_effect
                    print(f"Extracted media_effect with shape {media_effect.shape}")
    
    # Try to extract baseline
    if hasattr(model, 'baseline_geo_idx'):
        print("Found baseline_geo_idx")
        
        # Get posterior samples if not already defined
        if not 'posterior' in locals() and hasattr(model, 'posterior_samples'):
            posterior = model.posterior_samples
        
        # Try to calculate baseline if we have posterior samples
        if 'posterior' in locals() and isinstance(posterior, dict) and \
           'baseline_intercept' in posterior and 'baseline_trend' in posterior:
            print("Found baseline parameters in posterior")
            intercept = np.mean(posterior['baseline_intercept'], axis=0)
            trend = np.mean(posterior['baseline_trend'], axis=0)
            
            # Create time index
            time_idx = np.arange(len(actual_data))
            
            # Calculate baseline
            baseline = intercept + trend * time_idx
            components['baseline'] = baseline
            print(f"Calculated baseline with shape {baseline.shape}")
    
    # Try to extract seasonality
    if 'posterior' in locals() and isinstance(posterior, dict) and 'seasonality_features' in posterior:
        print("Found seasonality_features in posterior")
        seasonality = np.mean(posterior['seasonality_features'], axis=0)
        if len(seasonality.shape) > 1:
            seasonality = np.mean(seasonality, axis=0)
        components['seasonality'] = seasonality
        print(f"Extracted seasonality with shape {seasonality.shape}")
    
    # Check if we have any time-varying components
    time_varying_components = []
    for name, component in components.items():
        if len(component) == len(actual_data):
            time_varying_components.append(name)
    
    if time_varying_components:
        print(f"Found time-varying components: {', '.join(time_varying_components)}")
        
        # Create combined prediction
        prediction = np.zeros_like(actual_data)
        for name in time_varying_components:
            prediction += components[name]
        
        # If we have a baseline, add it
        if 'baseline' in components and len(components['baseline']) == len(actual_data):
            # Already included above
            pass
        else:
            # Add a constant offset to match the mean
            offset = np.mean(actual_data) - np.mean(prediction)
            prediction += offset
            print(f"Added constant offset: {offset}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create x values (simple indices)
        x = np.arange(len(actual_data))
        
        # Plot actual data
        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
        
        # Plot prediction
        plt.plot(x, prediction, '-', color='red', label='Model Prediction (Components)')
        
        # Plot individual components
        colors = ['green', 'purple', 'orange', 'cyan']
        for i, name in enumerate(time_varying_components):
            component = components[name]
            if len(component) == len(actual_data):
                plt.plot(x, component, '--', color=colors[i % len(colors)], 
                         alpha=0.5, label=f'{name.replace("_", " ").title()}')
        
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
        plt.title('Model Components vs Actual Data', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'model_components_vs_actual.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        print(f"Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
    else:
        print("Could not find any time-varying components")
        
        # Try a different approach - use the model's trace to extract predictions
        if hasattr(model, 'inference_data'):
            print("Found inference_data, trying to extract predictions from trace")
            
            try:
                import arviz as az
                
                # Get the inference data
                inference_data = model.inference_data
                
                # Print available groups and variables
                print(f"Available groups: {list(inference_data.groups())}")
                if 'posterior' in inference_data.groups():
                    print(f"Posterior variables: {list(inference_data.posterior.data_vars)}")
                
                # Try to extract mu_t (predicted values over time)
                if 'posterior' in inference_data.groups() and 'mu_t' in inference_data.posterior.data_vars:
                    print("Found 'mu_t' in posterior")
                    mu = inference_data.posterior['mu_t'].values
                elif 'posterior' in inference_data.groups() and 'mu' in inference_data.posterior.data_vars:
                    print("Found 'mu' in posterior")
                    mu = inference_data.posterior['mu'].values
                    
                    # Average across chains and draws
                    prediction = mu.mean(axis=(0, 1))
                    
                    # If prediction has multiple dimensions, flatten it
                    if len(prediction.shape) > 1:
                        prediction = prediction.reshape(-1)
                    
                    # If prediction length doesn't match actual data, try to reshape
                    if len(prediction) != len(actual_data):
                        print(f"Prediction length {len(prediction)} doesn't match actual data length {len(actual_data)}")
                        
                        # Try to reshape if prediction is longer
                        if len(prediction) > len(actual_data) and len(prediction) % len(actual_data) == 0:
                            factor = len(prediction) // len(actual_data)
                            prediction = prediction.reshape(-1, factor).mean(axis=1)
                            print(f"Reshaped prediction to length {len(prediction)}")
                        elif len(prediction) < len(actual_data) and len(actual_data) % len(prediction) == 0:
                            # Repeat prediction if it's shorter
                            factor = len(actual_data) // len(prediction)
                            prediction = np.repeat(prediction, factor)
                            print(f"Repeated prediction to length {len(prediction)}")
                    
                    if len(prediction) == len(actual_data):
                        # Create plot
                        plt.figure(figsize=(12, 8))
                        
                        # Create x values (simple indices)
                        x = np.arange(len(actual_data))
                        
                        # Plot actual data
                        plt.plot(x, actual_data, '-', color='blue', label='Actual Data')
                        plt.plot(x[::20], actual_data[::20], 'o', color='blue', markersize=5)
                        
                        # Plot prediction
                        plt.plot(x, prediction, '-', color='red', label='Model Prediction (from trace)')
                        
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
                        plt.title('Model Predictions vs Actual Data (from trace)', fontsize=14, fontweight='bold')
                        plt.xlabel('Time Period', fontsize=12)
                        plt.ylabel('Value', fontsize=12)
                        plt.grid(True, alpha=0.3, linestyle='--')
                        plt.legend(loc='upper right')
                        
                        # Save plot
                        output_path = os.path.join(OUTPUT_DIR, 'model_trace_vs_actual.png')
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Plot saved to: {output_path}")
                        print(f"Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
            except ImportError:
                print("Could not import arviz to extract trace")
            except Exception as e:
                print(f"Error extracting from inference_data: {e}")

if __name__ == "__main__":
    main()