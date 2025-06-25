#!/usr/bin/env python3
"""
Meridian Convergence Fix
=======================
Minimal fixes for MCMC convergence issues
"""

import numpy as np
import tensorflow_probability as tfp
from meridian import constants
from meridian.model import prior_distribution, spec

def create_better_priors():
    """Create more constrained priors for better convergence"""
    return prior_distribution.PriorDistribution(
        # More constrained ROI prior
        roi_m=tfp.distributions.LogNormal(
            loc=np.log(0.1),  # Lower expected ROI
            scale=0.2,        # Much tighter variance
            name=constants.ROI_M
        ),
        # Tighter saturation prior
        saturation_gamma=tfp.distributions.Beta(
            concentration1=2.0,
            concentration0=1.0,
            name=constants.SATURATION_GAMMA
        )
    )

def get_better_sampling_params():
    """Return better sampling parameters"""
    return {
        'n_chains': 2,        # Fewer chains
        'n_adapt': 2000,      # Much more adaptation
        'n_burnin': 2000,     # Much more burnin
        'n_keep': 500,        # Fewer samples
        'seed': 42,
        'target_accept_prob': 0.8  # Lower acceptance rate
    }

def preprocess_data(data):
    """Scale data to improve convergence"""
    # Scale media spend to reasonable range
    if hasattr(data, 'media_spend'):
        data.media_spend = data.media_spend / np.max(data.media_spend) * 100
    
    # Scale KPI if very large
    if hasattr(data, 'kpi') and np.max(data.kpi) > 1e6:
        data.kpi = data.kpi / 1000
    
    return data

# Usage example:
if __name__ == "__main__":
    print("Use these fixes in your model:")
    print("1. prior = create_better_priors()")
    print("2. data = preprocess_data(data)")
    print("3. mmm.sample_posterior(**get_better_sampling_params())")