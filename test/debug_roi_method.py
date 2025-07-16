#!/usr/bin/env python3
"""
Debug the ROI method shape mismatch issue
"""

import pickle
import numpy as np
import tensorflow as tf
from meridian.analysis import analyzer

def debug_roi_method():
    print("🔍 DEBUGGING ROI METHOD")
    print("=" * 50)
    
    # Load fitted model
    try:
        model_path = '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl'
        with open(model_path, 'rb') as f:
            mmm = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create analyzer
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # Inspect ROI method
    print("\nROI METHOD INSPECTION:")
    try:
        # Get method source if possible
        import inspect
        if hasattr(analyzer_obj, 'roi') and callable(analyzer_obj.roi):
            try:
                source = inspect.getsource(analyzer_obj.roi)
                print(f"ROI method source:\n{source[:500]}...")
            except:
                print("Could not get source code")
        
        # Try to get ROI with debug info
        print("\nTrying ROI method with debug info:")
        try:
            # Get shape of posterior samples
            if hasattr(mmm, 'posterior_samples'):
                samples = mmm.posterior_samples
                print(f"Posterior samples type: {type(samples)}")
                if hasattr(samples, 'posterior'):
                    print(f"Posterior keys: {list(samples.posterior.keys())[:5]}...")
                    
                    # Check shapes of key posterior parameters
                    for key in list(samples.posterior.keys())[:5]:
                        try:
                            shape = samples.posterior[key].shape
                            print(f"  {key}: {shape}")
                        except:
                            print(f"  {key}: shape unknown")
            else:
                print("No posterior_samples attribute")
            
            # Try ROI with use_kpi=True and debug=True
            print("\nCalling roi(use_kpi=True):")
            try:
                roi_results = analyzer_obj.roi(use_kpi=True)
                print(f"ROI results shape: {roi_results.shape}")
                print(f"ROI results type: {type(roi_results)}")
                print(f"First few values: {roi_results[:5]}")
            except Exception as roi_error:
                print(f"ROI error: {roi_error}")
                
                # Try to extract the error details
                import traceback
                print("\nDetailed error:")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Debug error: {e}")
    
    except Exception as e:
        print(f"Inspection failed: {e}")
    
    # Try manual ROI calculation
    print("\nMANUAL ROI CALCULATION:")
    try:
        # Get incremental contribution
        incremental = analyzer_obj.incremental_outcome(use_kpi=True)
        print(f"Incremental shape: {incremental.shape}")
        
        # Get total spend
        total_spend = np.sum(mmm.input_data.media_spend, axis=(0,1))
        print(f"Total spend shape: {total_spend.shape}")
        
        # Calculate channel contribution
        channel_contribution = [np.sum(incremental[:,:,i]) for i in range(incremental.shape[2])]
        print(f"Channel contribution shape: {np.array(channel_contribution).shape}")
        
        # Calculate ROI manually
        roi_values = []
        for i in range(len(channel_contribution)):
            if total_spend[i] > 0:
                roi = channel_contribution[i] / total_spend[i]
            else:
                roi = 0
            roi_values.append(roi)
        
        print(f"Manual ROI shape: {np.array(roi_values).shape}")
        print(f"Manual ROI values: {roi_values}")
        
    except Exception as e:
        print(f"Manual calculation failed: {e}")

if __name__ == "__main__":
    debug_roi_method()