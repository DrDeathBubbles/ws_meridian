#!/usr/bin/env python3
"""
Google Meridian Simple Tutorial Script
=====================================

A minimal working example of Google Meridian for Media Mix Modeling.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from meridian import constants
from meridian.data import load
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import analyzer
from meridian.analysis import visualizer

def main():
    print("=" * 50)
    print("GOOGLE MERIDIAN SIMPLE TUTORIAL")
    print("=" * 50)
    
    # Setup data loader with correct column names
    coord_to_columns = load.CoordToColumns(
        time='time',
        geo='geo',
        controls=['competitor_sales_control', 'sentiment_score_control'],
        population='population',
        kpi='conversions',
        revenue_per_kpi='revenue_per_conversion',
        media=[
            'Channel0_impression',
            'Channel1_impression', 
            'Channel2_impression',
            'Channel3_impression',
            'Channel4_impression',
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
            'Channel2_spend', 
            'Channel3_spend',
            'Channel4_spend',
        ],
        organic_media=['Organic_channel0_impression'],
        non_media_treatments=['Promo'],
    )

    media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1', 
        'Channel2_impression': 'Channel_2',
        'Channel3_impression': 'Channel_3',
        'Channel4_impression': 'Channel_4',
    }
    
    media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
        'Channel2_spend': 'Channel_2', 
        'Channel3_spend': 'Channel_3',
        'Channel4_spend': 'Channel_4',
    }

    # Load data
    print("Loading data...")
    loader = load.CsvDataLoader(
        csv_path="https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )
    
    data = loader.load()
    print("✓ Data loaded successfully!")
    print(f"KPI shape: {data.kpi.shape}")
    print(f"Media shape: {data.media.shape}")
    
    # Create model with better priors
    print("\nSetting up model...")
    roi_mu = np.log(0.5)  # More conservative ROI prior
    roi_sigma = 0.5       # Less variance
    
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    
    model_spec = spec.ModelSpec(prior=prior)
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    print("✓ Model initialized!")
    
    # Sample with conservative settings
    print("\nSampling from prior...")
    mmm.sample_prior(100)
    print("✓ Prior sampling complete!")
    
    print("\nTraining model (this will take a few minutes)...")
    try:
        mmm.sample_posterior(
            n_chains=2,      # Fewer chains
            n_adapt=1000,    # More adaptation
            n_burnin=1000,   # More burnin
            n_keep=500,      # Fewer samples
            seed=42
        )
        print("✓ Model training complete!")
        
        # Basic analysis
        print("\nPerforming analysis...")
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Get ROI estimates
        roi_estimates = analyzer_obj.get_roi()
        roi_mean = np.mean(roi_estimates, axis=0)
        
        print("\nROI Estimates by Channel:")
        for i, roi in enumerate(roi_mean):
            print(f"Channel_{i}: {roi:.3f}")
            
        # Try diagnostics (may still fail but won't crash)
        try:
            model_diagnostics = visualizer.ModelDiagnostics(mmm)
            plt.figure(figsize=(10, 6))
            model_diagnostics.plot_rhat_boxplot()
            plt.title("Model Diagnostics: R-hat Values")
            plt.savefig('model_diagnostics.png', dpi=150, bbox_inches='tight')
            print("✓ Diagnostics plot saved as model_diagnostics.png")
            plt.close()
        except Exception as e:
            print(f"Note: Diagnostics plot failed: {e}")
            print("This is common with tutorial data - model may need more tuning")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        print("This can happen with default settings. Try adjusting priors or sampling parameters.")
    
    print("\n" + "=" * 50)
    print("TUTORIAL COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()