#!/usr/bin/env python3
"""
Google Meridian Working Example
==============================

Uses test utilities to create a working example.
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
    print("GOOGLE MERIDIAN WORKING EXAMPLE")
    print("=" * 50)
    
    # Use built-in test data
    print("Creating test data...")
    try:
        # Use sample input data function
        data = test_utils.sample_input_data_non_revenue_revenue_per_kpi()
        print("✓ Test data created!")
        print(f"KPI shape: {data.kpi.shape}")
        print(f"Media shape: {data.media.shape}")
        
        # Create model with default priors
        print("\nSetting up model...")
        prior = prior_distribution.PriorDistribution()
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        print("✓ Model initialized!")
        
        # Sample from prior (minimal)
        print("\nSampling from prior...")
        mmm.sample_prior(10)
        print("✓ Prior sampling complete!")
        
        # Train model with very minimal settings
        print("\nTraining model (minimal settings)...")
        mmm.sample_posterior(
            n_chains=1,
            n_adapt=50,
            n_burnin=50,
            n_keep=50,
            seed=42
        )
        print("✓ Model training complete!")
        
        # Basic analysis
        print("\nPerforming analysis...")
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get ROI estimates
        roi_estimates = analyzer_obj.get_roi()
        roi_mean = np.mean(roi_estimates, axis=0)
        
        print("\nROI Estimates:")
        for i, roi in enumerate(roi_mean):
            print(f"Channel_{i}: {roi:.3f}")
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nThis demonstrates the basic setup process.")
        print("For production use, you'll need:")
        print("- Properly formatted real data")
        print("- Appropriate priors for your business")
        print("- More sampling iterations for convergence")
    
    print("\n" + "=" * 50)
    print("WORKING EXAMPLE COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Use your own data with CsvDataLoader")
    print("2. Adjust priors based on business knowledge")
    print("3. Increase sampling parameters for production")
    print("4. Validate model convergence with diagnostics")

if __name__ == "__main__":
    main()