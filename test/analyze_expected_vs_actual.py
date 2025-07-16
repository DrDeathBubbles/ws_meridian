#!/usr/bin/env python3
"""
Analyze Expected vs Actual
========================
Detailed analysis of expected vs actual values for both models
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from meridian.analysis import analyzer

def load_models():
    """Load both original and improved models"""
    models = {}
    
    # Load original model
    with open('/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl', 'rb') as f:
        models['original'] = pickle.load(f)
    
    # Load improved model
    with open('/Users/aaronmeagher/Work/google_meridian/google/test/improved_model/improved_model.pkl', 'rb') as f:
        models['improved'] = pickle.load(f)
    
    return models

def analyze_expected_vs_actual(models):
    """Detailed analysis of expected vs actual values"""
    print("\n📊 DETAILED EXPECTED VS ACTUAL ANALYSIS")
    
    # Create output directory
    import os
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Store data for both models
    data = {}
    
    # Extract data from both models
    for model_name, mmm in models.items():
        # Create analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get expected vs actual data
        expected_vs_actual = analyzer_obj.expected_vs_actual_data()
        
        # Extract data - handle different data structures
        if isinstance(expected_vs_actual, dict):
            # Dictionary format
            expected = expected_vs_actual.get('expected', [])
            actual = expected_vs_actual.get('actual', [])
            
            # Convert to 1D arrays if needed
            if hasattr(expected, 'shape') and len(expected.shape) > 1:
                expected = np.mean(expected, axis=tuple(range(len(expected.shape)-1)))
            if hasattr(actual, 'shape') and len(actual.shape) > 1:
                actual = np.mean(actual, axis=tuple(range(len(actual.shape)-1)))
                
            time_periods = range(len(expected))
        else:
            # Try to handle other formats
            if hasattr(expected_vs_actual, 'to_dataframe'):
                # Convert to DataFrame if possible
                df = expected_vs_actual.to_dataframe()
                if 'expected' in df.columns and 'actual' in df.columns:
                    expected = df['expected'].values
                    actual = df['actual'].values
                    time_periods = range(len(expected))
                else:
                    # Just use the first two columns
                    cols = df.columns[:2]
                    expected = df[cols[0]].values
                    actual = df[cols[1]].values
                    time_periods = range(len(expected))
            else:
                # Just use the raw data
                print(f"  ⚠ Unexpected data format for {model_name} model")
                continue
        
        # Store data
        data[model_name] = {
            'expected': expected,
            'actual': actual,
            'time_periods': time_periods
        }
    
    # Calculate metrics for both models
    metrics = {}
    
    for model_name, model_data in data.items():
        expected = model_data['expected']
        actual = model_data['actual']
        
        # Calculate correlation
        correlation = np.corrcoef(actual, expected)[0, 1]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((expected - actual) ** 2))
        
        # Calculate MAE
        mae = np.mean(np.abs(expected - actual))
        
        # Calculate R-squared
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - expected) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Store metrics
        metrics[model_name] = {
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared
        }
    
    # Print metrics
    print("\nMETRICS COMPARISON:")
    print(f"{'Metric':<15} {'Original':<15} {'Improved':<15} {'Difference':<15} {'Better Model':<15}")
    print("-" * 75)
    
    for metric in ['correlation', 'rmse', 'mae', 'r_squared']:
        orig = metrics['original'][metric]
        impr = metrics['improved'][metric]
        diff = impr - orig
        
        # For correlation and R-squared, higher is better; for RMSE and MAE, lower is better
        if metric in ['correlation', 'r_squared']:
            better = "Improved" if diff > 0 else "Original"
            diff_pct = diff / abs(orig) * 100 if orig != 0 else float('inf')
        else:
            better = "Improved" if diff < 0 else "Original"
            diff_pct = -diff / orig * 100 if orig != 0 else float('inf')
        
        print(f"{metric:<15} {orig:<15.3f} {impr:<15.3f} {diff_pct:<15.1f}% {better:<15}")
    
    # Create detailed plot
    plt.figure(figsize=(15, 12))
    
    # Plot both models in separate subplots
    for i, model_name in enumerate(['original', 'improved']):
        model_data = data[model_name]
        expected = model_data['expected']
        actual = model_data['actual']
        time_periods = model_data['time_periods']
        
        # Plot time series
        plt.subplot(3, 1, i+1)
        plt.plot(time_periods, actual, label='Actual', linewidth=2)
        plt.plot(time_periods, expected, label='Expected', linewidth=2, linestyle='--')
        
        # Calculate metrics
        correlation = metrics[model_name]['correlation']
        rmse = metrics[model_name]['rmse']
        r_squared = metrics[model_name]['r_squared']
        
        plt.title(f'{model_name.capitalize()} Model: Expected vs Actual\nCorrelation: {correlation:.3f}, RMSE: {rmse:.3f}, R²: {r_squared:.3f}', fontsize=14)
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('KPI Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot residuals (errors) for both models
    plt.subplot(3, 1, 3)
    
    for model_name, color in [('original', 'red'), ('improved', 'green')]:
        model_data = data[model_name]
        expected = model_data['expected']
        actual = model_data['actual']
        time_periods = model_data['time_periods']
        
        # Calculate residuals
        residuals = expected - actual
        
        # Plot residuals
        plt.plot(time_periods, residuals, label=f'{model_name.capitalize()} Residuals', 
                 color=color, alpha=0.7, linewidth=2)
    
    plt.title('Model Residuals (Expected - Actual)', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Residual Value', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    path = f"{output_dir}/expected_vs_actual_detailed.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: expected_vs_actual_detailed.png")
    
    # Create scatter plot for both models
    plt.figure(figsize=(15, 7))
    
    for i, model_name in enumerate(['original', 'improved']):
        model_data = data[model_name]
        expected = model_data['expected']
        actual = model_data['actual']
        
        # Plot scatter
        plt.subplot(1, 2, i+1)
        plt.scatter(actual, expected, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(np.min(actual), np.min(expected))
        max_val = max(np.max(actual), np.max(expected))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Calculate metrics
        correlation = metrics[model_name]['correlation']
        r_squared = metrics[model_name]['r_squared']
        
        plt.title(f'{model_name.capitalize()} Model: Actual vs Expected\nCorrelation: {correlation:.3f}, R²: {r_squared:.3f}', fontsize=14)
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Expected', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    path = f"{output_dir}/expected_vs_actual_scatter.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: expected_vs_actual_scatter.png")
    
    # Provide analysis summary
    print("\nANALYSIS SUMMARY:")
    
    # Determine which model is better overall
    better_count = sum(1 for metric in ['correlation', 'rmse', 'mae', 'r_squared'] 
                      if (metric in ['correlation', 'r_squared'] and metrics['improved'][metric] > metrics['original'][metric]) or
                         (metric in ['rmse', 'mae'] and metrics['improved'][metric] < metrics['original'][metric]))
    
    if better_count >= 3:
        print("✓ The improved model provides a better fit to the data overall.")
    else:
        print("✗ The original model provides a better fit to the data overall.")
    
    # Analyze correlation
    if metrics['improved']['correlation'] > metrics['original']['correlation']:
        print(f"✓ The improved model has a stronger correlation ({metrics['improved']['correlation']:.3f} vs {metrics['original']['correlation']:.3f}).")
    else:
        print(f"✗ The original model has a stronger correlation ({metrics['original']['correlation']:.3f} vs {metrics['improved']['correlation']:.3f}).")
    
    # Analyze R-squared
    if metrics['improved']['r_squared'] > metrics['original']['r_squared']:
        print(f"✓ The improved model explains more variance (R² = {metrics['improved']['r_squared']:.3f} vs {metrics['original']['r_squared']:.3f}).")
    else:
        print(f"✗ The original model explains more variance (R² = {metrics['original']['r_squared']:.3f} vs {metrics['improved']['r_squared']:.3f}).")
    
    # Analyze error metrics
    if metrics['improved']['rmse'] < metrics['original']['rmse']:
        print(f"✓ The improved model has lower prediction error (RMSE = {metrics['improved']['rmse']:.3f} vs {metrics['original']['rmse']:.3f}).")
    else:
        print(f"✗ The original model has lower prediction error (RMSE = {metrics['original']['rmse']:.3f} vs {metrics['improved']['rmse']:.3f}).")

def main():
    print("🔍 ANALYZING EXPECTED VS ACTUAL")
    print("=" * 50)
    
    # Load models
    models = load_models()
    
    # Analyze expected vs actual
    analyze_expected_vs_actual(models)
    
    print("\n✅ ANALYSIS COMPLETE")
    print("Check the model_evaluation directory for detailed visualizations.")

if __name__ == "__main__":
    main()