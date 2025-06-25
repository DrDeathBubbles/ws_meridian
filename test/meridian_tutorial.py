#!/usr/bin/env python3
"""
Google Meridian Tutorial Script
==============================

This tutorial demonstrates how to use Google Meridian for Media Mix Modeling (MMM).
It covers data loading, model specification, training, and analysis.

Requirements:
- meridian package installed
- tensorflow, tensorflow-probability
- pandas, numpy, matplotlib
- arviz for diagnostics
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az
import matplotlib.pyplot as plt
from psutil import virtual_memory

# Meridian imports
from meridian import constants
from meridian.data import load
from meridian.data import test_utils
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import optimizer
from meridian.analysis import analyzer
from meridian.analysis import visualizer
from meridian.analysis import summarizer
from meridian.analysis import formatter

def check_system_resources():
    """Check available system resources."""
    ram_gb = virtual_memory().total / 1e9
    print(f'Your runtime has {ram_gb:.1f} gigabytes of available RAM')
    print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print(f"Num CPUs Available: {len(tf.config.experimental.list_physical_devices('CPU'))}")
    print("-" * 50)

def setup_data_loader():
    """Set up the data loader with proper column mappings."""
    print("Setting up data loader...")
    
    # Define column mappings (corrected to match actual data)
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

    # Media to channel mappings
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

    # Create data loader
    loader = load.CsvDataLoader(
        csv_path="https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )
    
    return loader

def load_and_inspect_data(loader):
    """Load data and perform basic inspection."""
    print("Loading data...")
    try:
        data = loader.load()
        print("✓ Data loaded successfully!")
        
        # Basic data inspection
        print(f"KPI shape: {data.kpi.shape}")
        print(f"Media shape: {data.media.shape}")
        print(f"Time periods: {data.kpi.shape[0]}")
        print(f"Geos: {data.kpi.shape[1]}")
        print(f"Media channels: {data.media.shape[2]}")
        if hasattr(data, 'controls') and data.controls is not None:
            print(f"Controls shape: {data.controls.shape}")
        else:
            print("No controls data")
        
        return data
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def setup_model_spec():
    """Set up model specification with priors."""
    print("Setting up model specification...")
    
    # Define ROI priors
    roi_mu = 0.2     # Mean for ROI prior
    roi_sigma = 0.9  # Standard deviation for ROI prior
    
    # Create prior distribution
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    
    # Create model specification
    model_spec = spec.ModelSpec(prior=prior)
    print("✓ Model specification created!")
    
    return model_spec

def train_model(data, model_spec):
    """Train the Meridian model."""
    print("Initializing Meridian model...")
    
    # Create model
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    print("✓ Model initialized!")
    
    # Sample from prior (for validation)
    print("Sampling from prior...")
    mmm.sample_prior(500)
    print("✓ Prior sampling complete!")
    
    # Sample from posterior
    print("Training model (sampling posterior)...")
    print("This may take several minutes...")
    mmm.sample_posterior(
        n_chains=4,      # Reduced for faster execution
        n_adapt=500,
        n_burnin=500, 
        n_keep=1000,
        seed=42
    )
    print("✓ Model training complete!")
    
    return mmm

def analyze_model(mmm):
    """Perform model analysis and diagnostics."""
    print("Performing model diagnostics...")
    
    # Model diagnostics
    model_diagnostics = visualizer.ModelDiagnostics(mmm)
    
    # Plot R-hat diagnostics
    plt.figure(figsize=(10, 6))
    model_diagnostics.plot_rhat_boxplot()
    plt.title("Model Diagnostics: R-hat Values")
    plt.tight_layout()
    plt.savefig('test/model_diagnostics_rhat.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # Get media contribution
    print("Calculating media contributions...")
    media_contribution = analyzer_obj.get_media_contribution()
    print("Media contribution shape:", media_contribution.shape)
    
    # Get ROI estimates
    print("Calculating ROI estimates...")
    roi_estimates = analyzer_obj.get_roi()
    print("ROI estimates shape:", roi_estimates.shape)
    
    return analyzer_obj

def run_optimization(mmm):
    """Run budget optimization."""
    print("Running budget optimization...")
    
    try:
        # Create optimizer
        opt = optimizer.Optimizer(mmm)
        
        # Run optimization with current budget
        current_budget = mmm.input_data.media_spend.sum(axis=(0, 1))
        print(f"Current total budget: {current_budget.sum():.2f}")
        
        # Optimize budget allocation
        optimal_allocation = opt.optimize_budget(
            budget=current_budget.sum(),
            n_time_periods=4  # Optimize for next 4 periods
        )
        
        print("✓ Budget optimization complete!")
        return optimal_allocation
        
    except Exception as e:
        print(f"Note: Budget optimization failed: {e}")
        return None

def main():
    """Main tutorial execution."""
    print("=" * 60)
    print("GOOGLE MERIDIAN TUTORIAL")
    print("=" * 60)
    
    # Check system resources
    check_system_resources()
    
    # Set up data loader
    loader = setup_data_loader()
    
    # Load and inspect data
    data = load_and_inspect_data(loader)
    if data is None:
        print("Tutorial stopped due to data loading error.")
        return
    
    # Set up model specification
    model_spec = setup_model_spec()
    
    # Train model
    mmm = train_model(data, model_spec)
    
    # Analyze model
    analyzer_obj = analyze_model(mmm)
    
    # Run optimization
    optimal_allocation = run_optimization(mmm)
    
    print("=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("Files created:")
    print("- test/model_diagnostics_rhat.png")
    print("\nNext steps:")
    print("- Review model diagnostics")
    print("- Analyze media contributions")
    print("- Experiment with different priors")
    print("- Apply to your own data")

if __name__ == "__main__":
    main()