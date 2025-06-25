#!/usr/bin/env python3
"""
TensorFlow dtype workaround for Meridian
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution

# Force TensorFlow to use float64 globally
tf.keras.backend.set_floatx('float64')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    try:
        # Load data
        coord_to_columns = load.CoordToColumns(
            time='date_id',
            geo='national_geo',
            kpi='clicks',
            media_spend=['google_spend', 'facebook_spend', 'linkedin_spend'],
            media=['google_impressions', 'facebook_impressions', 'linkedin_impressions'],
        )
        
        media_to_channel = {
            'google_impressions': 'Google',
            'facebook_impressions': 'Facebook',
            'linkedin_impressions': 'LinkedIn'
        }
        
        media_spend_to_channel = {
            'google_spend': 'Google',
            'facebook_spend': 'Facebook',
            'linkedin_spend': 'LinkedIn'
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
        
        # Minimal prior - use defaults
        prior = prior_distribution.PriorDistribution()
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        print("✓ Model created")
        
        # Try minimal sampling
        mmm.sample_prior(5)
        print("✓ Prior sampling worked!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nKnown GitHub issues:")
        print("1. https://github.com/google/meridian/issues - Check for dtype issues")
        print("2. https://github.com/tensorflow/probability/issues - TFP compatibility")
        print("3. Try downgrading TensorFlow: pip install tensorflow==2.13.0")

if __name__ == "__main__":
    main()