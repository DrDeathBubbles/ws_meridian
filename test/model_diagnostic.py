#!/usr/bin/env python3
"""
Model Diagnostic
===============
Check what's actually saved in the fitted model
"""

import pickle
import numpy as np

def diagnose_model():
    """Check model contents"""
    print("🔍 MODEL DIAGNOSTIC")
    print("=" * 50)
    
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl', 'rb') as f:
            mmm = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return
    
    # Check model attributes
    print("\nMODEL ATTRIBUTES:")
    attrs = [attr for attr in dir(mmm) if not attr.startswith('_')]
    for attr in attrs[:10]:  # First 10 attributes
        try:
            value = getattr(mmm, attr)
            if hasattr(value, 'shape'):
                print(f"  {attr}: {type(value).__name__} {value.shape}")
            elif callable(value):
                print(f"  {attr}: {type(value).__name__} (method)")
            else:
                print(f"  {attr}: {type(value).__name__}")
        except:
            print(f"  {attr}: Access failed")
    
    # Check for posterior samples
    print(f"\nPOSTERIOR SAMPLES CHECK:")
    if hasattr(mmm, 'posterior_samples'):
        samples = mmm.posterior_samples
        if samples is None:
            print("  ⚠ posterior_samples is None")
        else:
            print(f"  ✓ posterior_samples exists: {type(samples)}")
            
            # Check samples attributes
            sample_attrs = [attr for attr in dir(samples) if not attr.startswith('_')]
            print(f"  Sample attributes: {sample_attrs[:5]}...")
    else:
        print("  ✗ No posterior_samples attribute")
    
    # Check for other sample storage
    for attr_name in ['_posterior_samples', 'samples', 'mcmc_samples', 'trace']:
        if hasattr(mmm, attr_name):
            attr_value = getattr(mmm, attr_name)
            print(f"  Found {attr_name}: {type(attr_value)}")
    
    # Check input data
    print(f"\nINPUT DATA:")
    if hasattr(mmm, 'input_data'):
        data = mmm.input_data
        print(f"  KPI shape: {data.kpi.shape}")
        print(f"  Media shape: {data.media.shape}")
        print(f"  Media spend shape: {data.media_spend.shape}")
        print(f"  Total KPI: {np.sum(data.kpi):,.0f}")
        print(f"  Total spend: €{np.sum(data.media_spend):,.0f}")
    
    print(f"\n📋 SUMMARY:")
    print("The model contains input data but may not have")
    print("posterior samples saved properly. This is common")
    print("with some Meridian versions.")
    
    print(f"\n💡 WHAT YOU CAN STILL ANALYZE:")
    print("• Model structure and parameters")
    print("• Input data patterns and relationships") 
    print("• Data preprocessing results")
    print("• Model setup validation")

if __name__ == "__main__":
    diagnose_model()