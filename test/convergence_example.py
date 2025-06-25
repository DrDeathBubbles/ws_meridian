#!/usr/bin/env python3
"""
Example with convergence fixes applied
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution

# Fix TensorFlow dtype issues
tf.config.experimental.enable_tensor_float_32_execution(False)

# Your data loading code (replace with your actual data)
def load_your_data():
    coord_to_columns = load.CoordToColumns(
        time='date_id',
        geo='national_geo',
        kpi='clicks',  # Using clicks as KPI since no conversions column
        media_spend=['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 'bing_spend'],
        media=['google_impressions', 'facebook_impressions', 'linkedin_impressions', 'reddit_impressions', 'bing_impressions'],
    )
    
    # Required mappings
    media_to_channel = {
        'google_impressions': 'Google',
        'facebook_impressions': 'Facebook', 
        'linkedin_impressions': 'LinkedIn',
        'reddit_impressions': 'Reddit',
        'bing_impressions': 'Bing'
    }
    
    media_spend_to_channel = {
        'google_spend': 'Google',
        'facebook_spend': 'Facebook',
        'linkedin_spend': 'LinkedIn', 
        'reddit_spend': 'Reddit',
        'bing_spend': 'Bing'
    }
    
    loader = load.CsvDataLoader(
        csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data.csv',
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )
    
    return loader.load()

def main():
    try:
        # Load data
        data = load_your_data()
        print("✓ Data loaded successfully")
        
        # Convert to float64 to fix dtype issues
        data.kpi = data.kpi.astype(np.float64)
        if hasattr(data, 'media'):
            data.media = data.media.astype(np.float64)
        if hasattr(data, 'media_spend'):
            data.media_spend = data.media_spend.astype(np.float64)
        
        # Scale data
        if np.max(data.kpi) > 1e6:
            data.kpi = data.kpi / 1000
        
        if hasattr(data, 'media_spend') and np.max(data.media_spend) > 0:
            data.media_spend = data.media_spend / np.max(data.media_spend) * 100
    
        # Better priors with float64
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(
                loc=tf.constant(np.log(0.1), dtype=tf.float64), 
                scale=tf.constant(0.2, dtype=tf.float64), 
                name=constants.ROI_M
            )
        )
        
        # Create model
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        print("✓ Model created successfully")
        
        # Sample prior first
        print("Sampling prior...")
        mmm.sample_prior(10)
        print("✓ Prior sampling complete")
        
        # Sample with better parameters
        print("Training model...")
        mmm.sample_posterior(
            n_chains=1,  # Start with 1 chain
            n_adapt=100,  # Reduce for testing
            n_burnin=100,
            n_keep=100,
            seed=42
        )
        print("✓ Model training complete")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This demonstrates the setup process - adjust data and parameters as needed")

if __name__ == "__main__":
    main()