#!/usr/bin/env python3
"""
Train Model and Create Prediction Plot
=====================================
Trains the improved model and creates a plot comparing actual training data 
against model predictions with confidence intervals.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from improved_model_implementation import fit_improved_model

# Set output directory
OUTPUT_DIR = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'

def create_prediction_plot(model):
    """Create plot comparing actual data vs predictions with confidence intervals"""
    print("\nCreating prediction plot...")
    
    # Get the actual data (KPI values)
    actual_data = model.input_data.kpi_df
    
    # Get model predictions with confidence intervals
    predictions = model.predict_kpi()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
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
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('KPI Value (Clicks)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    # Calculate and display metrics
    mse = np.mean((actual_data.values - predictions['mean'].values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_data.values - predictions['mean'].values))
    
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
    print("🔍 TRAINING MODEL AND GENERATING PREDICTION PLOT")
    print("=" * 50)
    
    # Train model (or load if already trained)
    try:
        # First try to load the model
        import pickle
        with open(f'{OUTPUT_DIR}/improved_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded from existing file")
    except (FileNotFoundError, ImportError):
        # If loading fails, train a new model
        print("Training new model...")
        model = fit_improved_model()
    
    # Create prediction plot
    mse, rmse, mae = create_prediction_plot(model)
    
    print("\n✅ PREDICTION PLOT GENERATED")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"\nCheck {OUTPUT_DIR}/model_predictions_vs_actual.png for the visualization.")

if __name__ == "__main__":
    main()