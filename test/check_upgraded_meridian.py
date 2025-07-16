#!/usr/bin/env python3
"""
Check Upgraded Meridian
=====================
Verify if the upgraded Meridian version supports variable slope parameters
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from meridian import constants
from meridian.model import prior_distribution

def check_prior_capabilities():
    """Check if the prior distribution supports custom slope parameters"""
    print("Checking prior distribution capabilities...")
    
    # Try to create a prior with custom slope parameters
    try:
        # Create a custom slope distribution
        slope_dist = tfp.distributions.LogNormal(
            loc=np.log(2.0) - 0.25/2,  # Mean of 2.0
            scale=0.25,
            name='custom_slope'
        )
        
        # Try to create a prior with the custom slope
        prior = prior_distribution.PriorDistribution(slope_m=slope_dist)
        print("✓ Successfully created prior with custom slope parameter")
        print("  This version of Meridian supports variable slope parameters!")
        return True
    except Exception as e:
        print(f"✗ Failed to create prior with custom slope: {e}")
        print("  This version of Meridian may not support variable slope parameters")
        return False

def check_meridian_version():
    """Check the installed Meridian version"""
    try:
        import meridian
        print(f"Meridian version: {meridian.__version__}")
    except Exception as e:
        print(f"Error checking Meridian version: {e}")
    
    try:
        import google_meridian
        print(f"Google Meridian version: {google_meridian.__version__}")
    except Exception as e:
        print(f"Error checking Google Meridian version: {e}")

def main():
    print("🔍 CHECKING UPGRADED MERIDIAN")
    print("=" * 50)
    
    # Check Meridian version
    check_meridian_version()
    
    # Check prior capabilities
    supports_variable_slope = check_prior_capabilities()
    
    print("\n✅ CHECK COMPLETE")
    if supports_variable_slope:
        print("The upgraded version of Meridian supports variable slope parameters.")
        print("You can now create a model with proper S-shaped hill curves.")
    else:
        print("The upgraded version of Meridian may not support variable slope parameters.")
        print("You may need to use a different approach or contact Google Meridian support.")

if __name__ == "__main__":
    main()