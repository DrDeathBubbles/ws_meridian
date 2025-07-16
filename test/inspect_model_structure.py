#!/usr/bin/env python3
"""
Inspect Model Structure
======================
Examine the structure of the fitted model to find parameters
"""

import pickle
import numpy as np
import pandas as pd

def load_model():
    """Load the fitted model"""
    model_paths = [
        '/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl',
        '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl',
        '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Moderate_model.pkl'
    ]
    
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                mmm = pickle.load(f)
            print(f"✓ Model loaded successfully from {path}")
            return mmm, path
        except Exception as e:
            print(f"✗ Failed to load model from {path}: {e}")
    
    print("No models could be loaded. Please check file paths.")
    return None, None

def inspect_model_structure(mmm, model_path):
    """Inspect the structure of the model object"""
    print("\n📊 MODEL STRUCTURE INSPECTION")
    
    # Get all attributes
    attrs = dir(mmm)
    non_private_attrs = [attr for attr in attrs if not attr.startswith('_')]
    
    print(f"Model has {len(non_private_attrs)} non-private attributes:")
    for attr in sorted(non_private_attrs):
        try:
            value = getattr(mmm, attr)
            if hasattr(value, 'shape'):
                print(f"  • {attr}: {type(value).__name__} with shape {value.shape}")
            elif isinstance(value, (list, tuple, dict)):
                print(f"  • {attr}: {type(value).__name__} with length {len(value)}")
            elif callable(value):
                print(f"  • {attr}: {type(value).__name__} (callable)")
            else:
                print(f"  • {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"  • {attr}: Error accessing - {e}")
    
    # Look for potential parameter storage
    print("\nSearching for parameter storage:")
    parameter_containers = ['posterior_samples', 'samples', 'mcmc_samples', 'trace', 
                           'params', 'parameters', 'model_params', 'fitted_params']
    
    for container in parameter_containers:
        if hasattr(mmm, container):
            container_obj = getattr(mmm, container)
            print(f"\nFound parameter container: {container}")
            
            if container_obj is None:
                print("  Container is None")
                continue
                
            # Try to inspect the container
            if hasattr(container_obj, '__dict__'):
                container_attrs = dir(container_obj)
                non_private_container_attrs = [attr for attr in container_attrs if not attr.startswith('_')]
                print(f"  Container has {len(non_private_container_attrs)} non-private attributes:")
                
                for attr in sorted(non_private_container_attrs[:20]):  # Show first 20
                    try:
                        value = getattr(container_obj, attr)
                        if hasattr(value, 'shape'):
                            print(f"    - {attr}: {type(value).__name__} with shape {value.shape}")
                        elif isinstance(value, (list, tuple, dict)):
                            print(f"    - {attr}: {type(value).__name__} with length {len(value)}")
                        else:
                            print(f"    - {attr}: {type(value).__name__}")
                    except Exception as e:
                        print(f"    - {attr}: Error accessing - {e}")
                
                if len(non_private_container_attrs) > 20:
                    print(f"    ... and {len(non_private_container_attrs) - 20} more attributes")
            elif isinstance(container_obj, dict):
                print(f"  Container is a dictionary with {len(container_obj)} keys:")
                for key in list(container_obj.keys())[:20]:  # Show first 20
                    value = container_obj[key]
                    if hasattr(value, 'shape'):
                        print(f"    - {key}: {type(value).__name__} with shape {value.shape}")
                    elif isinstance(value, (list, tuple, dict)):
                        print(f"    - {key}: {type(value).__name__} with length {len(value)}")
                    else:
                        print(f"    - {key}: {type(value).__name__}")
                
                if len(container_obj) > 20:
                    print(f"    ... and {len(container_obj) - 20} more keys")
    
    # Check for model_spec which might contain prior information
    if hasattr(mmm, 'model_spec'):
        print("\nExamining model_spec:")
        model_spec = mmm.model_spec
        
        if hasattr(model_spec, 'prior'):
            print("  Found prior specification:")
            prior = model_spec.prior
            
            # Try to extract prior information
            prior_attrs = dir(prior)
            non_private_prior_attrs = [attr for attr in prior_attrs if not attr.startswith('_')]
            
            for attr in non_private_prior_attrs:
                try:
                    value = getattr(prior, attr)
                    print(f"    - {attr}: {type(value).__name__}")
                    
                    # If it's a distribution, try to get parameters
                    if 'distribution' in str(type(value)).lower():
                        if hasattr(value, 'parameters'):
                            params = value.parameters
                            print(f"      Parameters: {params}")
                        elif hasattr(value, 'loc') and hasattr(value, 'scale'):
                            print(f"      loc: {value.loc}, scale: {value.scale}")
                except Exception as e:
                    print(f"    - {attr}: Error accessing - {e}")
    
    # Check for analyzer which might have access to parameters
    from meridian.analysis import analyzer
    try:
        analyzer_obj = analyzer.Analyzer(mmm)
        print("\nCreated analyzer object to access parameters")
        
        # Try to get adstock parameters
        try:
            adstock_decay = analyzer_obj.adstock_decay()
            print("  ✓ Got adstock_decay data")
            
            if isinstance(adstock_decay, pd.DataFrame):
                print(f"  Columns: {adstock_decay.columns.tolist()}")
                
                # Check for half-life column
                if 'half_life' in adstock_decay.columns:
                    print("  Found half-life column!")
                    
                    # Group by channel and get mean half-life
                    if 'channel' in adstock_decay.columns:
                        half_lives = adstock_decay.groupby('channel')['half_life'].mean()
                        print("\nActual half-life values by channel:")
                        for channel, half_life in half_lives.items():
                            print(f"  • {channel}: {half_life:.2f} days")
                        
                        # Save to file
                        half_lives.to_csv('/Users/aaronmeagher/Work/google_meridian/google/test/actual_half_lives.csv')
                        print("  ✓ Saved half-lives to actual_half_lives.csv")
        except Exception as e:
            print(f"  ✗ Error getting adstock_decay: {e}")
        
        # Try to get hill curve parameters
        try:
            hill_curves = analyzer_obj.hill_curves()
            print("  ✓ Got hill_curves data")
            
            if isinstance(hill_curves, pd.DataFrame):
                print(f"  Columns: {hill_curves.columns.tolist()}")
                
                # Check for ec50 and slope columns
                if 'ec50' in hill_curves.columns and 'slope' in hill_curves.columns:
                    print("  Found ec50 and slope columns!")
                    
                    # Group by channel and get mean parameters
                    if 'channel' in hill_curves.columns:
                        params = hill_curves.groupby('channel')[['ec50', 'slope']].mean()
                        print("\nActual hill curve parameters by channel:")
                        for channel, row in params.iterrows():
                            print(f"  • {channel}: EC50={row['ec50']:.2f}, Slope={row['slope']:.2f}")
                        
                        # Save to file
                        params.to_csv('/Users/aaronmeagher/Work/google_meridian/google/test/actual_hill_params.csv')
                        print("  ✓ Saved hill parameters to actual_hill_params.csv")
        except Exception as e:
            print(f"  ✗ Error getting hill_curves: {e}")
    except Exception as e:
        print(f"✗ Error creating analyzer: {e}")

def main():
    print("🔍 INSPECTING MODEL STRUCTURE")
    print("=" * 50)
    
    # Load model
    mmm, model_path = load_model()
    if mmm is None:
        return
    
    # Inspect model structure
    inspect_model_structure(mmm, model_path)
    
    print("\n✅ MODEL INSPECTION COMPLETE")

if __name__ == "__main__":
    main()