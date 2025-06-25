#!/usr/bin/env python3
"""
Full working Meridian example with workarounds
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution

# TensorFlow fixes
tf.keras.backend.set_floatx('float64')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    try:
        # Load data with fewer channels to avoid issues
        coord_to_columns = load.CoordToColumns(
            time='date_id',
            geo='national_geo',
            kpi='clicks',
            media_spend=['google_spend', 'facebook_spend'],  # Only 2 channels
            media=['google_impressions', 'facebook_impressions'],
        )
        
        media_to_channel = {
            'google_impressions': 'Google',
            'facebook_impressions': 'Facebook'
        }
        
        media_spend_to_channel = {
            'google_spend': 'Google',
            'facebook_spend': 'Facebook'
        }
        
        loader = load.CsvDataLoader(
            csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data.csv',
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=media_to_channel,
            media_spend_to_channel=media_spend_to_channel,
        )
        
        data = loader.load()
        print("✓ Data loaded")
        
        # Use default priors to avoid dtype issues
        prior = prior_distribution.PriorDistribution()
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        print("✓ Model created")
        
        # Prior sampling
        mmm.sample_prior(5)
        print("✓ Prior sampling complete")
        
        # Posterior sampling with minimal settings
        print("Starting posterior sampling...")
        mmm.sample_posterior(
            n_chains=1,
            n_adapt=50,
            n_burnin=50,
            n_keep=50,
            seed=42
        )
        print("✓ Posterior sampling complete!")
        
        # Basic analysis
        from meridian.analysis import analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        media_contribution = analyzer_obj.get_media_contribution()
        print(f"✓ Analysis complete: Media contribution shape {media_contribution.shape}")
        
        # Get summary
        print(f"Total media contribution: {np.sum(media_contribution):.0f}")
        print(f"Media contribution by channel: {np.mean(media_contribution, axis=(0,1))}")
        
        print("\n🎉 SUCCESS! Meridian model fitted and analyzed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting links:")
        print("• GitHub Issues: https://github.com/google/meridian/issues")
        print("• TensorFlow compatibility: https://github.com/tensorflow/tensorflow/issues")
        print("• Try: pip install tensorflow==2.13.0 tensorflow-probability==0.21.0")

if __name__ == "__main__":
    main()