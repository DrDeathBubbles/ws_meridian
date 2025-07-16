#!/usr/bin/env python3
"""
Posterior Sample Analysis
========================
Extract confidence intervals from fitted Meridian model
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_posterior_samples():
    """Analyze posterior samples for statistical confidence"""
    print("📊 POSTERIOR SAMPLE ANALYSIS")
    print("=" * 50)
    
    # Load fitted model
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl', 'rb') as f:
            mmm = pickle.load(f)
    except:
        print("No fitted model found. Run meridian_success_grid_2.py first.")
        return
    
    # Access posterior samples
    if hasattr(mmm, 'posterior_samples') and mmm.posterior_samples is not None:
        samples = mmm.posterior_samples
        print(f"✓ Posterior samples found")
        
        # Sample info
        if hasattr(samples, 'n_chains') and hasattr(samples, 'n_keep'):
            total_samples = samples.n_chains * samples.n_keep
            print(f"Total samples: {total_samples:,}")
            print(f"Chains: {samples.n_chains}, Samples per chain: {samples.n_keep}")
        
        # Access posterior parameters
        if hasattr(samples, 'posterior'):
            posterior_vars = list(samples.posterior.keys())
            print(f"Parameters: {posterior_vars[:5]}...")
            
            # Calculate confidence intervals
            print("\nCONFIDENCE INTERVALS (90%):")
            for param_name in posterior_vars[:5]:  # First 5 parameters
                try:
                    param_samples = samples.posterior[param_name]
                    param_flat = param_samples.flatten() if hasattr(param_samples, 'flatten') else param_samples
                    
                    p5 = np.percentile(param_flat, 5)
                    p50 = np.percentile(param_flat, 50)
                    p95 = np.percentile(param_flat, 95)
                    
                    print(f"{param_name}: {p50:.3f} [{p5:.3f}, {p95:.3f}]")
                except:
                    print(f"{param_name}: Analysis failed")
            
            # Create posterior plots
            plot_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors'
            
            plt.figure(figsize=(15, 10))
            n_params = min(6, len(posterior_vars))
            
            for i, param_name in enumerate(posterior_vars[:n_params]):
                plt.subplot(2, 3, i+1)
                try:
                    param_samples = samples.posterior[param_name]
                    param_flat = param_samples.flatten() if hasattr(param_samples, 'flatten') else param_samples
                    
                    plt.hist(param_flat, bins=30, alpha=0.7, density=True)
                    plt.title(f'{param_name}')
                    plt.xlabel('Value')
                    plt.ylabel('Density')
                    
                    # Add confidence interval lines
                    p5 = np.percentile(param_flat, 5)
                    p95 = np.percentile(param_flat, 95)
                    plt.axvline(p5, color='red', linestyle='--', alpha=0.7, label='90% CI')
                    plt.axvline(p95, color='red', linestyle='--', alpha=0.7)
                    
                except:
                    plt.text(0.5, 0.5, 'Analysis\nFailed', ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.tight_layout()
            path = f'{plot_dir}/Posterior_Distributions.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved: Posterior_Distributions.png")
            
        else:
            print("⚠ Posterior structure not accessible")
    else:
        print("⚠ No posterior samples available")

if __name__ == "__main__":
    analyze_posterior_samples()