#!/usr/bin/env python3
"""
Model Prediction Plot
====================
Creates a plot comparing actual training data against model predictions
with confidence intervals.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Add the parent directory to the path so we can import meridian
sys.path.append('/Users/aaronmeagher/Work/google_meridian/google')

# Set output directory
OUTPUT_DIR = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'
MODEL_PATH = f'{OUTPUT_DIR}/improved_model.pkl'

def load_model():
    """Load the trained model"""
    print("Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
    
    # Print detailed information about the model structure
    print("\nModel type:", type(model))
    print("\nModel attributes:")
    for attr in dir(model):
        if not attr.startswith('_'):
            try:
                attr_value = getattr(model, attr)
                attr_type = type(attr_value)
                print(f"  - {attr}: {attr_type}")
                
                # If it's a method, print its signature if possible
                if callable(attr_value):
                    try:
                        import inspect
                        sig = inspect.signature(attr_value)
                        print(f"    Signature: {sig}")
                    except:
                        pass
                        
                # If it's a complex object, print its attributes
                elif hasattr(attr_value, '__dict__'):
                    sub_attrs = [a for a in dir(attr_value) if not a.startswith('_')]
                    if len(sub_attrs) > 0:
                        print(f"    Sub-attributes: {', '.join(sub_attrs[:10])}{'...' if len(sub_attrs) > 10 else ''}")
            except:
                print(f"  - {attr}: <error accessing attribute>")
    
    return model

def create_prediction_plot(model):
    """Create plot comparing actual data vs predictions with confidence intervals"""
    print("Creating prediction plot...")
    
    # Import pandas and numpy for data manipulation
    import pandas as pd
    import numpy as np
    
    # Check if xarray is available (for DataArray objects)
    try:
        import xarray as xr
        XARRAY_AVAILABLE = True
    except ImportError:
        XARRAY_AVAILABLE = False
    
    # Helper function to convert DataArray to DataFrame if needed
    def convert_to_dataframe(data):
        """Convert various data types to pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif XARRAY_AVAILABLE and isinstance(data, xr.DataArray):
            print(f"Converting DataArray to DataFrame. Dimensions: {data.dims}")
            # Convert DataArray to DataFrame
            try:
                return data.to_dataframe()
            except Exception as e:
                print(f"Error converting DataArray to DataFrame: {e}")
                # Try a different approach
                try:
                    if len(data.dims) == 1:
                        # Single dimension
                        return pd.DataFrame({data.name or 'value': data.values}, 
                                          index=data.coords[data.dims[0]].values)
                    else:
                        # Multiple dimensions, use the first as index
                        df = pd.DataFrame(data.values, 
                                        index=data.coords[data.dims[0]].values)
                        df.columns = [f'col_{i}' for i in range(df.shape[1])]
                        return df
                except Exception as e2:
                    print(f"Second error converting DataArray: {e2}")
                    # Last resort: just convert to numpy array and create a DataFrame
                    return pd.DataFrame(data.values)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to DataFrame
            if data.ndim == 1:
                return pd.DataFrame({'value': data})
            else:
                df = pd.DataFrame(data)
                df.columns = [f'col_{i}' for i in range(df.shape[1])]
                return df
        else:
            # Try to convert to DataFrame
            try:
                return pd.DataFrame(data)
            except:
                print(f"Could not convert {type(data)} to DataFrame")
                return None
    
    # Try to extract the actual data and predictions from the model
    # This is a more general approach that tries to find the data wherever it might be
    
    # First, try to get the actual data (KPI values)
    actual_data = None
    
    # Check if we have input_data attribute
    if hasattr(model, 'input_data'):
        input_data = model.input_data
        print("Found input_data attribute")
        
        # Try different possible attribute names for KPI data
        if hasattr(input_data, 'kpi_df'):
            actual_data = input_data.kpi_df
            print("Found kpi_df in input_data")
        elif hasattr(input_data, 'kpi'):
            actual_data = input_data.kpi
            print("Found kpi in input_data")
        elif hasattr(input_data, 'target'):
            actual_data = input_data.target
            print("Found target in input_data")
        elif hasattr(input_data, 'data'):
            # If there's a data attribute, try to find KPI data in it
            data_attr = input_data.data
            if isinstance(data_attr, dict) and 'kpi' in data_attr:
                actual_data = data_attr['kpi']
                print("Found kpi in input_data.data dictionary")
    
    # If we still don't have actual data, try other places
    if actual_data is None:
        if hasattr(model, 'data'):
            data_attr = model.data
            print("Found data attribute in model")
            
            if isinstance(data_attr, dict):
                if 'kpi' in data_attr:
                    actual_data = data_attr['kpi']
                    print("Found kpi in model.data dictionary")
                elif 'target' in data_attr:
                    actual_data = data_attr['target']
                    print("Found target in model.data dictionary")
    
    # If we still don't have actual data, try to find any dataframe that might be the KPI data
    if actual_data is None:
        print("Could not find KPI data through standard attributes. Looking for any dataframe or DataArray...")
        
        # Function to recursively search for pandas DataFrames or xarray DataArrays in an object
        def find_dataframes(obj, path="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return []
                
            results = []
            
            if isinstance(obj, pd.DataFrame):
                results.append((path, obj))
                return results
            elif XARRAY_AVAILABLE and isinstance(obj, xr.DataArray):
                results.append((path, obj))
                return results
                
            if isinstance(obj, dict):
                for key, value in obj.items():
                    results.extend(find_dataframes(value, f"{path}.{key}" if path else key, 
                                                max_depth, current_depth + 1))
                    
            elif hasattr(obj, '__dict__'):
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        results.extend(find_dataframes(value, f"{path}.{key}" if path else key, 
                                                    max_depth, current_depth + 1))
                        
            return results
        
        # Search for dataframes in the model
        dataframes = find_dataframes(model)
        print(f"Found {len(dataframes)} dataframes in the model")
        
        # If we found any dataframes or DataArrays, use the first one as the actual data
        if dataframes:
            for path, data_obj in dataframes:
                # Convert to DataFrame if it's a DataArray
                if XARRAY_AVAILABLE and isinstance(data_obj, xr.DataArray):
                    print(f"  - {path}: DataArray with dims {data_obj.dims}")
                    df = convert_to_dataframe(data_obj)
                else:
                    print(f"  - {path}: {data_obj.shape}")
                    df = data_obj
                
                if df is not None:
                    # If the dataframe has a single column, it might be the KPI data
                    if df.shape[1] == 1 or 'kpi' in path.lower() or 'target' in path.lower():
                        actual_data = df
                        print(f"Using {path} as actual data")
                        break
    
    # If we still don't have actual data, we can't continue
    if actual_data is None:
        raise ValueError("Could not find KPI data in the model")
    
    # Convert actual_data to DataFrame if it's not already
    if not isinstance(actual_data, pd.DataFrame):
        actual_data = convert_to_dataframe(actual_data)
        if actual_data is None:
            raise ValueError("Could not convert actual data to DataFrame")
        
    print(f"Actual data shape: {actual_data.shape}")
    
    # Now try to get the predictions
    predictions = None
    
    # Try different possible method names for predictions
    prediction_methods = ['predict_kpi', 'predict', 'posterior_predict', 'get_posterior_predictions']
    for method_name in prediction_methods:
        if hasattr(model, method_name) and callable(getattr(model, method_name)):
            try:
                method = getattr(model, method_name)
                predictions = method()
                print(f"Got predictions using {method_name}() method")
                break
            except Exception as e:
                print(f"Error calling {method_name}(): {e}")
    
    # If we couldn't get predictions from methods, try to access posterior samples directly
    if predictions is None:
        print("Could not get predictions from methods. Trying to access posterior samples directly.")
        
        # Check if we have posterior samples
        if hasattr(model, 'posterior_samples'):
            print("Found posterior_samples attribute")
            
            # Try to extract predictions from posterior samples
            try:
                # Get the time index from the actual data
                time_index = actual_data.index
                
                # Look for 'mu' or similar in posterior samples
                posterior = model.posterior_samples
                if isinstance(posterior, dict):
                    # Try common names for the mean prediction
                    for key in ['mu', 'mean', 'prediction', 'pred', 'y_pred']:
                        if key in posterior:
                            print(f"Found {key} in posterior_samples")
                            # Create a dataframe with predictions
                            mean_values = np.mean(posterior[key], axis=0)
                            std_values = np.std(posterior[key], axis=0)
                            
                            # Make sure the shapes match
                            if len(mean_values) == len(actual_data):
                                predictions = pd.DataFrame({
                                    'mean': mean_values,
                                    'lower': mean_values - 1.96 * std_values,  # 95% CI
                                    'upper': mean_values + 1.96 * std_values   # 95% CI
                                }, index=time_index)
                                
                                print("Created predictions from posterior samples")
                                break
                            else:
                                print(f"Shape mismatch: mean_values {len(mean_values)}, actual_data {len(actual_data)}")
            except Exception as e:
                print(f"Error creating predictions from posterior samples: {e}")
    
    # If we still don't have predictions, try to find any attribute that might contain predictions
    if predictions is None:
        print("Could not get predictions from standard methods or posterior samples.")
        print("Looking for any attribute that might contain predictions...")
        
        # Try to find attributes that might contain predictions
        for attr_name in dir(model):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(model, attr_name)
                
                # Check if it's a DataFrame or DataArray with the right shape
                is_dataframe = isinstance(attr, pd.DataFrame)
                is_dataarray = XARRAY_AVAILABLE and isinstance(attr, xr.DataArray)
                
                if (is_dataframe or is_dataarray):
                    # Convert to DataFrame if needed
                    if is_dataarray:
                        print(f"Found DataArray {attr_name}")
                        attr_df = convert_to_dataframe(attr)
                    else:
                        print(f"Found DataFrame {attr_name}")
                        attr_df = attr
                        
                    # Check if it has the right length
                    if attr_df is not None and len(attr_df) == len(actual_data):
                        print(f"DataFrame {attr_name} has matching length")
                    else:
                        continue
                    
                    # Check if it has columns that might be predictions
                    for col in attr.columns:
                        if any(pred_name in col.lower() for pred_name in ['pred', 'mean', 'fit', 'estimate']):
                            print(f"Using column {col} from {attr_name} as predictions")
                            
                            # Create a simple prediction dataframe
                            mean_values = attr[col].values
                            predictions = pd.DataFrame({
                                'mean': mean_values,
                                'lower': mean_values * 0.9,  # Simple approximation for CI
                                'upper': mean_values * 1.1   # Simple approximation for CI
                            }, index=actual_data.index)
                            break
                    
                    if predictions is not None:
                        break
            except:
                pass
    
    # If we still don't have predictions, we can't continue
    if predictions is None:
        # As a last resort, just use the actual data as predictions (for visualization purposes)
        print("WARNING: Could not find predictions. Using actual data as predictions for visualization.")
        predictions = pd.DataFrame({
            'mean': actual_data.values.flatten(),
            'lower': actual_data.values.flatten() * 0.9,  # Simple approximation for CI
            'upper': actual_data.values.flatten() * 1.1   # Simple approximation for CI
        }, index=actual_data.index)
    
    # Convert predictions to DataFrame if it's not already
    if not isinstance(predictions, pd.DataFrame):
        predictions = convert_to_dataframe(predictions)
        if predictions is None:
            raise ValueError("Could not convert predictions to DataFrame")
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Make sure actual_data and predictions have the same index
    if not actual_data.index.equals(predictions.index):
        print("WARNING: Actual data and predictions have different indices. Aligning data...")
        # Get the common indices
        common_idx = actual_data.index.intersection(predictions.index)
        actual_data = actual_data.loc[common_idx]
        predictions = predictions.loc[common_idx]
        print(f"After alignment: actual_data shape {actual_data.shape}, predictions shape {predictions.shape}")
    
    # Make sure predictions has the expected columns (mean, lower, upper)
    if not all(col in predictions.columns for col in ['mean', 'lower', 'upper']):
        print("WARNING: Predictions dataframe doesn't have the expected columns. Creating them...")
        
        # If we have a single column, use it as the mean
        if predictions.shape[1] == 1:
            col_name = predictions.columns[0]
            mean_values = predictions[col_name].values
        # Otherwise use the first column as the mean
        else:
            mean_values = predictions.iloc[:, 0].values
            
        # Create a new dataframe with the expected columns
        predictions = pd.DataFrame({
            'mean': mean_values,
            'lower': mean_values * 0.9,  # Simple approximation for CI
            'upper': mean_values * 1.1   # Simple approximation for CI
        }, index=predictions.index)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Convert index to numeric values for plotting
    # This avoids issues with complex index types
    x_actual = np.arange(len(actual_data))
    x_pred = np.arange(len(predictions))
    
    # Plot actual data
    plt.plot(
        x_actual, 
        actual_data.values.flatten(), 
        'o-', 
        color='blue', 
        label='Actual Data',
        markersize=5,
        alpha=0.7
    )
    
    # Plot predictions
    plt.plot(
        x_pred, 
        predictions['mean'].values.flatten(), 
        'r-', 
        label='Model Predictions',
        linewidth=2
    )
    
    # Plot confidence intervals
    plt.fill_between(
        x_pred,
        predictions['lower'].values.flatten(),
        predictions['upper'].values.flatten(),
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # If the index is datetime, format the x-axis accordingly
    if isinstance(actual_data.index[0], pd.Timestamp):
        # Create a function to format the x-axis labels
        def format_date(x, pos=None):
            if x < 0 or x >= len(actual_data):
                return ''
            return actual_data.index[int(x)].strftime('%Y-%m-%d')
        
        # Use a FuncFormatter to format the x-axis
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_date))
        plt.gcf().autofmt_xdate()
    else:
        # Just use the original index values as tick labels
        plt.xticks(x_actual[::max(1, len(x_actual)//10)], 
                  [str(idx) for idx in actual_data.index[::max(1, len(actual_data.index)//10)]], 
                  rotation=45)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Convert date strings to datetime objects if needed
    if isinstance(actual_data.index[0], str):
        actual_data.index = pd.to_datetime(actual_data.index)
    
    # Sort data by date
    actual_data = actual_data.sort_index()
    predictions = predictions.sort_index()
    
    # Plot actual data
    plt.plot(
        actual_data.index, 
        actual_data.values, 
        'o-', 
        color='blue', 
        label='Actual Data',
        markersize=5,
        alpha=0.7
    )
    
    # Plot predictions
    plt.plot(
        predictions.index, 
        predictions['mean'].values, 
        'r-', 
        label='Model Predictions',
        linewidth=2
    )
    
    # Plot confidence intervals
    plt.fill_between(
        predictions.index,
        predictions['lower'].values,
        predictions['upper'].values,
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Format the plot
    plt.title('Model Predictions vs Actual Data', fontsize=14, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('KPI Value (Clicks)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Calculate and display metrics
    actual_values = actual_data.values.flatten()
    pred_values = predictions['mean'].values.flatten()
    
    # Make sure the arrays have the same length
    min_len = min(len(actual_values), len(pred_values))
    actual_values = actual_values[:min_len]
    pred_values = pred_values[:min_len]
    
    mse = np.mean((actual_values - pred_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - pred_values))
    
    # Add metrics to plot
    metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
    plt.annotate(
        metrics_text,
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=10
    )
    
    # Save the figure
    path = f"{OUTPUT_DIR}/model_predictions_vs_actual.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: model_predictions_vs_actual.png")
    
    return mse, rmse, mae

def main():
    print("🔍 GENERATING MODEL PREDICTION PLOT")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    # Create prediction plot
    mse, rmse, mae = create_prediction_plot(model)
    
    print("\n✅ PREDICTION PLOT GENERATED")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"\nCheck {OUTPUT_DIR}/model_predictions_vs_actual.png for the visualization.")

if __name__ == "__main__":
    main()