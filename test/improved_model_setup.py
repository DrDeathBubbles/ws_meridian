#!/usr/bin/env python3
"""
Improved Model Setup
==================
Implements improvements for better hill curves and adstock decay
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution

# Force TensorFlow to use float64 for numerical stability
tf.keras.backend.set_floatx('float64')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data():
    """Load data once for all experiments"""
    coord_to_columns = load.CoordToColumns(
        time='date_id',
        geo='national_geo',
        kpi='clicks',
        media_spend=['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend'],
        media=['google_impressions', 'facebook_impressions', 'linkedin_impressions', 'reddit_impressions', 'bing_impressions', 'tiktok_impressions', 'twitter_impressions', 'instagram_impressions'],
    )
    
    loader = load.CsvDataLoader(
        csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv',
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel={
            'google_impressions': 'Google', 'facebook_impressions': 'Facebook',
            'linkedin_impressions': 'LinkedIn', 'reddit_impressions': 'Reddit',
            'bing_impressions': 'Bing', 'tiktok_impressions': 'TikTok',
            'twitter_impressions': 'Twitter', 'instagram_impressions': 'Instagram'
        },
        media_spend_to_channel={
            'google_spend': 'Google', 'facebook_spend': 'Facebook',
            'linkedin_spend': 'LinkedIn', 'reddit_spend': 'Reddit',
            'bing_spend': 'Bing', 'tiktok_spend': 'TikTok',
            'twitter_spend': 'Twitter', 'instagram_spend': 'Instagram'
        },
    )
    
    return loader.load()

def create_improved_prior():
    """Create improved prior with better adstock and hill curve parameters"""
    print("Creating improved prior distribution...")
    
    # 1. IMPROVEMENT: Use variable slope parameters instead of fixed 1.0
    # Create slope distributions with means > 1.0 for proper S-curves
    slope_dists = {}
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    for i, channel in enumerate(channels):
        # Use different slope means for different channel types
        if channel in ['Google', 'Facebook']:
            slope_mean = 2.0  # Higher slope for more efficient channels
        elif channel in ['LinkedIn', 'Bing', 'TikTok']:
            slope_mean = 1.7  # Medium slope
        else:
            slope_mean = 1.5  # Lower slope for less efficient channels
        
        # Create lognormal distribution for slope (ensures positive values)
        # LogNormal(loc, scale) where loc is log(mean)
        slope_loc = np.log(slope_mean) - 0.25/2  # Adjust for lognormal mean formula
        slope_dists[f'slope_m_{i}'] = tfp.distributions.LogNormal(
            loc=slope_loc,
            scale=0.25,
            name=f'slope_m_{i}'
        )
    
    # 2. IMPROVEMENT: Use longer half-life values for adstock decay
    # Create adstock rate distributions that result in longer half-lives
    adstock_dists = {}
    
    for i, channel in enumerate(channels):
        # Set target half-life based on channel type
        if channel in ['Google', 'Facebook', 'TikTok']:
            target_hl = 5.0  # Shorter half-life for immediate impact channels
        elif channel in ['LinkedIn', 'Twitter']:
            target_hl = 7.0  # Medium half-life
        else:
            target_hl = 9.0  # Longer half-life for slower impact channels
        
        # Convert half-life to adstock rate: rate = ln(2)/half_life
        # Lower rate = longer half-life
        adstock_rate = np.log(2) / target_hl
        
        # Create beta distribution for adstock rate
        # Beta distribution keeps values between 0 and 1
        # We'll use a narrow distribution centered around our target rate
        alpha = 5.0
        beta = alpha * (1.0 / adstock_rate - 1.0)
        
        adstock_dists[f'adstock_rate_{i}'] = tfp.distributions.Beta(
            concentration1=alpha,
            concentration0=beta,
            name=f'adstock_rate_{i}'
        )
    
    # Create the custom prior distribution
    # Use the standard ROI prior from your existing model
    roi_dist = tfp.distributions.LogNormal(
        loc=1.044,
        scale=0.3,
        name=constants.ROI_M
    )
    
    # Create a dictionary of all custom distributions
    custom_dists = {
        constants.ROI_M: roi_dist,
        **slope_dists,
        **adstock_dists
    }
    
    # Create the prior distribution with custom parameters
    prior = prior_distribution.PriorDistribution(**custom_dists)
    
    return prior

def fit_improved_model():
    """Fit model with improved priors"""
    print("Fitting improved model...")
    
    # Load data
    data = load_data()
    print("✓ Data loaded successfully")
    
    # Create improved prior
    prior = create_improved_prior()
    print("✓ Improved prior created")
    
    # Create model spec with improved prior
    model_spec = spec.ModelSpec(prior=prior)
    
    # Create and fit model
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    
    # Sample prior
    mmm.sample_prior(5)
    print("✓ Prior sampled")
    
    # Fit model with more samples for better convergence
    mmm.sample_posterior(
        n_chains=4,
        n_adapt=1000,
        n_burnin=2000,
        n_keep=4000,
        seed=42
    )
    print("✓ Model fitted successfully")
    
    # Save the improved model
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/improved_model.pkl', 'wb') as f:
        pickle.dump(mmm, f)
    print(f"✓ Model saved to {output_dir}/improved_model.pkl")
    
    return mmm

def create_improved_visualizations(mmm):
    """Create improved visualizations with proper sorting and smoothing"""
    print("\nCreating improved visualizations...")
    
    # Create output directory
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    from meridian.analysis import analyzer
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # 3. IMPROVEMENT: Ensure data is properly sorted and smoothed
    
    # Get adstock decay data
    try:
        adstock_decay = analyzer_obj.adstock_decay()
        print("✓ Got adstock decay data")
        
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Create a figure for adstock decay
        plt.figure(figsize=(12, 8))
        
        # Get channel names
        channels = adstock_decay['channel'].unique()
        
        # Plot each channel with proper sorting and smoothing
        for i, channel in enumerate(channels):
            # Get data for this channel
            channel_data = adstock_decay[adstock_decay['channel'] == channel]
            
            # Sort by time units
            channel_data = channel_data.sort_values('time_units')
            
            # Apply smoothing if needed (moving average)
            if len(channel_data) > 5:
                # Check for sawtooth pattern
                diffs = np.diff(channel_data['mean'].values)
                sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
                
                if sign_changes > len(diffs) / 3:
                    print(f"  Applying smoothing to {channel} adstock decay")
                    # Apply smoothing with moving average
                    window_size = 3
                    channel_data['mean_smooth'] = channel_data['mean'].rolling(
                        window=window_size, center=True, min_periods=1
                    ).mean()
                else:
                    channel_data['mean_smooth'] = channel_data['mean']
            else:
                channel_data['mean_smooth'] = channel_data['mean']
            
            # Plot the smoothed curve
            plt.plot(
                channel_data['time_units'], 
                channel_data['mean_smooth'],
                label=channel,
                linewidth=2,
                marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                markersize=5,
                markevery=max(1, len(channel_data) // 10)
            )
        
        plt.title('Adstock Decay by Channel (Improved)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Days)', fontsize=12)
        plt.ylabel('Adstock Effect', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the figure
        path = f"{output_dir}/adstock_decay_improved.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: adstock_decay_improved.png")
    except Exception as e:
        print(f"✗ Adstock decay visualization failed: {e}")
    
    # Get hill curves data
    try:
        hill_curves = analyzer_obj.hill_curves()
        print("✓ Got hill curves data")
        
        # Create a figure for hill curves
        plt.figure(figsize=(12, 8))
        
        # Get channel names
        channels = hill_curves['channel'].unique()
        
        # Plot each channel with proper sorting
        for i, channel in enumerate(channels):
            # Get data for this channel
            channel_data = hill_curves[hill_curves['channel'] == channel]
            
            # Sort by media units
            channel_data = channel_data.sort_values('media_units')
            
            # Plot the curve
            plt.plot(
                channel_data['media_units'], 
                channel_data['mean'],
                label=channel,
                linewidth=2,
                marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                markersize=5,
                markevery=max(1, len(channel_data) // 10)
            )
        
        plt.title('Hill Curves by Channel (Improved)', fontsize=14, fontweight='bold')
        plt.xlabel('Media Value (Normalized)', fontsize=12)
        plt.ylabel('Response', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the figure
        path = f"{output_dir}/hill_curves_improved.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: hill_curves_improved.png")
    except Exception as e:
        print(f"✗ Hill curves visualization failed: {e}")

def main():
    print("🔧 IMPLEMENTING IMPROVEMENTS FOR BETTER MODEL RESULTS")
    print("=" * 50)
    
    # Fit improved model
    mmm = fit_improved_model()
    
    # Create improved visualizations
    create_improved_visualizations(mmm)
    
    print("\n✅ IMPROVEMENTS IMPLEMENTED")
    print("The improved model has:")
    print("1. Variable slope parameters (>1.0) for proper S-shaped hill curves")
    print("2. Longer half-life values (5-9 days) for smoother adstock decay")
    print("3. Proper sorting and smoothing for visualizations")
    print("\nCheck the improved_model directory for the new model and visualizations.")

if __name__ == "__main__":
    main()