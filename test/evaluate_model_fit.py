#!/usr/bin/env python3
"""
Evaluate Model Fit
================
Compare the original and improved models to determine parameter accuracy
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
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl', 'rb') as f:
            models['original'] = pickle.load(f)
        print("✓ Original model loaded")
    except Exception as e:
        print(f"✗ Failed to load original model: {e}")
    
    # Load improved model
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/improved_model/improved_model.pkl', 'rb') as f:
            models['improved'] = pickle.load(f)
        print("✓ Improved model loaded")
    except Exception as e:
        print(f"✗ Failed to load improved model: {e}")
    
    return models

def compare_model_fit(models):
    """Compare model fit metrics between original and improved models"""
    print("\n📊 COMPARING MODEL FIT")
    
    results = {}
    
    for model_name, mmm in models.items():
        # Create analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get predictive accuracy
        try:
            accuracy = analyzer_obj.predictive_accuracy()
            results[model_name] = accuracy
            print(f"\n{model_name.upper()} MODEL ACCURACY:")
            print(f"  • R-Squared: {accuracy['r_squared']:.3f}")
            print(f"  • MAPE: {accuracy['mape']:.3f}")
            print(f"  • wMAPE: {accuracy['wmape']:.3f}")
        except Exception as e:
            print(f"✗ Failed to get accuracy for {model_name} model: {e}")
    
    # Compare results
    if 'original' in results and 'improved' in results:
        print("\nCOMPARISON:")
        for metric in ['r_squared', 'mape', 'wmape']:
            if metric in results['original'] and metric in results['improved']:
                orig = results['original'][metric]
                impr = results['improved'][metric]
                diff = impr - orig
                
                # For R-squared, higher is better; for MAPE and wMAPE, lower is better
                if metric == 'r_squared':
                    better = diff > 0
                    diff_pct = diff / abs(orig) * 100 if orig != 0 else float('inf')
                else:
                    better = diff < 0
                    diff_pct = -diff / orig * 100 if orig != 0 else float('inf')
                
                status = "✓ Better" if better else "✗ Worse"
                print(f"  • {metric}: {status} by {abs(diff_pct):.1f}%")
    
    return results

def compare_expected_vs_actual(models):
    """Compare expected vs actual values for both models"""
    print("\n📈 COMPARING EXPECTED VS ACTUAL")
    
    # Create output directory
    import os
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot expected vs actual for both models
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, mmm) in enumerate(models.items()):
        # Create analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get expected vs actual data
        try:
            expected_vs_actual = analyzer_obj.expected_vs_actual_data()
            
            # Plot in subplot
            plt.subplot(2, 1, i+1)
            
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
            
            # Plot expected vs actual
            plt.plot(time_periods, actual, label='Actual', linewidth=2)
            plt.plot(time_periods, expected, label='Expected', linewidth=2, linestyle='--')
            
            # Calculate correlation
            correlation = np.corrcoef(actual, expected)[0, 1]
            
            plt.title(f'{model_name.capitalize()} Model: Expected vs Actual (Correlation: {correlation:.3f})', fontsize=14)
            plt.xlabel('Time Period', fontsize=12)
            plt.ylabel('KPI Value', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            print(f"  • {model_name.capitalize()} model correlation: {correlation:.3f}")
            
        except Exception as e:
            print(f"✗ Failed to plot expected vs actual for {model_name} model: {e}")
    
    plt.tight_layout()
    
    # Save the figure
    path = f"{output_dir}/expected_vs_actual_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: expected_vs_actual_comparison.png")

def compare_adstock_parameters(models):
    """Compare adstock parameters between models"""
    print("\n⏱️ COMPARING ADSTOCK PARAMETERS")
    
    for model_name, mmm in models.items():
        # Create analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get adstock decay data
        try:
            adstock_decay = analyzer_obj.adstock_decay()
            
            # Calculate half-life for each channel
            if isinstance(adstock_decay, pd.DataFrame) and 'channel' in adstock_decay.columns:
                channels = adstock_decay['channel'].unique()
                
                print(f"\n{model_name.upper()} MODEL HALF-LIVES:")
                for channel in channels:
                    # Get data for this channel
                    channel_data = adstock_decay[adstock_decay['channel'] == channel].sort_values('time_units')
                    
                    if len(channel_data) > 5:
                        # Find the time where the effect is closest to 0.5
                        half_effect_idx = np.abs(channel_data['mean'].values - 0.5).argmin()
                        half_life = channel_data['time_units'].values[half_effect_idx]
                        
                        print(f"  • {channel}: {half_life:.2f} days")
        except Exception as e:
            print(f"✗ Failed to get adstock parameters for {model_name} model: {e}")

def main():
    print("🔍 EVALUATING MODEL FIT")
    print("=" * 50)
    
    # Load models
    models = load_models()
    
    # Compare model fit
    compare_model_fit(models)
    
    # Compare expected vs actual
    compare_expected_vs_actual(models)
    
    # Compare adstock parameters
    compare_adstock_parameters(models)
    
    print("\n✅ EVALUATION COMPLETE")
    print("To determine if the parameters accurately reflect your data:")
    print("1. Compare the R-squared, MAPE, and wMAPE metrics")
    print("2. Check the correlation between expected and actual values")
    print("3. Examine the half-life values to see if they make sense for your channels")
    print("4. Look at the expected vs actual plots to see which model better captures the data patterns")
    print("\nA model with better fit metrics and higher correlation indicates")
    print("that its parameters more accurately reflect your data.")

if __name__ == "__main__":
    main()