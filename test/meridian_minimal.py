#!/usr/bin/env python3
"""
Google Meridian Minimal Working Example
======================================

Uses built-in test data to demonstrate basic functionality.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from meridian import constants
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import analyzer

def main():
    print("=" * 50)
    print("GOOGLE MERIDIAN MINIMAL EXAMPLE")
    print("=" * 50)
    
    # Use built-in test data
    print("Generating test data...")
    data = test_utils.generate_test_data(
        n_time_periods=52,  # 1 year of weekly data
        n_geos=10,          # 10 geographic regions
        n_media_channels=3, # 3 media channels
        n_controls=2,       # 2 control variables
        seed=42
    )
    
    print("✓ Test data generated!")
    print(f"KPI shape: {data.kpi.shape}")
    print(f"Media shape: {data.media.shape}")
    print(f"Media spend shape: {data.media_spend.shape}")
    
    # Create model with simple priors
    print("\nSetting up model...")
    prior = prior_distribution.PriorDistribution()  # Use defaults
    model_spec = spec.ModelSpec(prior=prior)
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    print("✓ Model initialized!")
    
    # Sample from prior
    print("\nSampling from prior...")
    mmm.sample_prior(50)
    print("✓ Prior sampling complete!")
    
    # Train model with minimal settings
    print("\nTraining model...")
    try:
        mmm.sample_posterior(
            n_chains=2,
            n_adapt=100,
            n_burnin=100,
            n_keep=100,
            seed=42
        )
        print("✓ Model training complete!")
        
        # Basic analysis
        print("\nPerforming analysis...")
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get media contribution
        media_contribution = analyzer_obj.get_media_contribution()
        print(f"Media contribution shape: {media_contribution.shape}")
        
        # Get ROI estimates
        roi_estimates = analyzer_obj.get_roi()
        roi_mean = np.mean(roi_estimates, axis=0)
        
        print("\nROI Estimates by Channel:")
        for i, roi in enumerate(roi_mean):
            print(f"Channel_{i}: {roi:.3f}")
            
        print("\nMedia Contribution Summary:")
        contrib_mean = np.mean(media_contribution, axis=(0, 1))
        for i, contrib in enumerate(contrib_mean):
            print(f"Channel_{i}: {contrib:.1f}")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        print("This can happen due to convergence issues with minimal data.")
    
    print("\n" + "=" * 50)
    print("MINIMAL EXAMPLE COMPLETE!")
    print("=" * 50)
    print("\nKey takeaways:")
    print("- Data loading and model setup work correctly")
    print("- Use test_utils.generate_test_data() for experimentation")
    print("- Adjust sampling parameters for better convergence")
    print("- Real data typically works better than synthetic test data")

if __name__ == "__main__":
    main()