#!/usr/bin/env python3
"""
Meridian Full Analysis Grid Search
=================================
Grid search with complete MMM analysis using all available methods
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from meridian import constants
from meridian.data import load
from meridian.model import model, spec, prior_distribution
from meridian.analysis import analyzer

# KEY FIX: Force TensorFlow to use float64
tf.keras.backend.set_floatx('float64')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data():
    """Load data once for all experiments"""
    coord_to_columns = load.CoordToColumns(
        time='date_id',
        geo='national_geo',
        kpi='clicks',
        media_spend=['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend'],
        media=['google_impressions', 'facebook_impressions', 'linkedin_impressions', 'reddit_impressions', 'bing_impressions', 'tiktok_impressions', 'twitter_impressions', 'instagram_impressions'],
    )
    
    loader = load.CsvDataLoader(
        csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv',
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel={
            'google_impressions': 'Google', 'facebook_impressions': 'Facebook',
            'linkedin_impressions': 'LinkedIn', 'reddit_impressions': 'Reddit',
            'bing_impressions': 'Bing', 'tiktok_impressions': 'TikTok',
            'twitter_impressions': 'Twitter', 'instagram_impressions': 'Instagram'
        },
        media_spend_to_channel={
            'google_spend': 'Google', 'facebook_spend': 'Facebook',
            'linkedin_spend': 'LinkedIn', 'reddit_spend': 'Reddit',
            'bing_spend': 'Bing', 'tiktok_spend': 'TikTok',
            'twitter_spend': 'Twitter', 'instagram_spend': 'Instagram'
        },
    )
    
    return loader.load()

def create_prior_configs():
    """Data-driven prior configurations"""
    configs = [
        {
            'name': 'Conservative',
            'roi_loc': 1.044,
            'roi_scale': 0.3,
        },
        {
            'name': 'Moderate',
            'roi_loc': 1.737,
            'roi_scale': 0.5,
        }
    ]
    return configs

def perform_full_analysis(mmm, config_name):
    """Perform complete MMM analysis using all available methods"""
    print(f"   🔬 Running full analysis for {config_name}...")
    
    analyzer_obj = analyzer.Analyzer(mmm)
    analysis_results = {}
    
    # 1. Incremental outcome (media contribution)
    try:
        incremental = analyzer_obj.incremental_outcome()
        analysis_results['incremental_outcome'] = {
            'shape': incremental.shape,
            'total_incremental': float(np.sum(incremental)),
            'by_channel': [float(np.sum(incremental[:,:,i])) for i in range(incremental.shape[2])]
        }
        print(f"      ✓ Incremental outcome: {np.sum(incremental):,.0f} total")
    except Exception as e:
        analysis_results['incremental_outcome'] = f"Failed: {e}"
    
    # 2. Baseline summary
    try:
        baseline = analyzer_obj.baseline_summary_metrics()
        analysis_results['baseline_summary'] = baseline
        print(f"      ✓ Baseline analysis complete")
    except Exception as e:
        analysis_results['baseline_summary'] = f"Failed: {e}"
    
    # 3. ROI analysis
    try:
        roi_results = analyzer_obj.roi()
        analysis_results['roi'] = {
            'shape': roi_results.shape,
            'mean_roi': [float(np.mean(roi_results[:,i])) for i in range(roi_results.shape[1])]
        }
        print(f"      ✓ ROI analysis: {len(roi_results[0])} channels")
    except Exception as e:
        analysis_results['roi'] = f"Failed: {e}"
    
    # 4. Response curves (KPI-based)
    try:
        response = analyzer_obj.response_curves(use_kpi=True)
        analysis_results['response_curves'] = f"Available: {type(response)}"
        print(f"      ✓ Response curves generated")
    except Exception as e:
        analysis_results['response_curves'] = f"Failed: {e}"
    
    # 5. Summary metrics
    try:
        summary = analyzer_obj.summary_metrics()
        analysis_results['summary_metrics'] = summary
        print(f"      ✓ Summary metrics complete")
    except Exception as e:
        analysis_results['summary_metrics'] = f"Failed: {e}"
    
    # 6. R-hat diagnostics
    try:
        rhat = analyzer_obj.get_rhat()
        analysis_results['rhat'] = {
            'max_rhat': float(np.max(rhat)) if hasattr(rhat, 'shape') else str(rhat),
            'mean_rhat': float(np.mean(rhat)) if hasattr(rhat, 'shape') else str(rhat)
        }
        print(f"      ✓ R-hat diagnostics complete")
    except Exception as e:
        analysis_results['rhat'] = f"Failed: {e}"
    
    return analysis_results

def create_analysis_plots(mmm, config_name, analysis_results):
    """Create plots from analysis results"""
    plot_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors_2'
    os.makedirs(plot_dir, exist_ok=True)
    
    analyzer_obj = analyzer.Analyzer(mmm)
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    plots_created = []
    
    # 1. Incremental contribution by channel
    try:
        if 'incremental_outcome' in analysis_results and 'by_channel' in analysis_results['incremental_outcome']:
            contrib_by_channel = analysis_results['incremental_outcome']['by_channel']
            
            plt.figure(figsize=(12, 6))
            plt.bar(channels[:len(contrib_by_channel)], contrib_by_channel)
            plt.title(f'Incremental Contribution by Channel - {config_name}')
            plt.xlabel('Channel')
            plt.ylabel('Incremental Clicks')
            plt.xticks(rotation=45)
            
            path = f'{plot_dir}/{config_name}_01_incremental_contribution.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(f'{config_name}_01_incremental_contribution.png')
    except Exception as e:
        print(f"      ⚠ Incremental plot failed: {e}")
    
    # 2. ROI by channel
    try:
        if 'roi' in analysis_results and 'mean_roi' in analysis_results['roi']:
            roi_by_channel = analysis_results['roi']['mean_roi']
            
            plt.figure(figsize=(12, 6))
            plt.bar(channels[:len(roi_by_channel)], roi_by_channel)
            plt.title(f'ROI by Channel - {config_name}')
            plt.xlabel('Channel')
            plt.ylabel('ROI')
            plt.xticks(rotation=45)
            
            path = f'{plot_dir}/{config_name}_02_roi_by_channel.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_created.append(f'{config_name}_02_roi_by_channel.png')
    except Exception as e:
        print(f"      ⚠ ROI plot failed: {e}")
    
    # 3. Response curves (KPI-based)
    try:
        response_curves = analyzer_obj.response_curves(use_kpi=True)
        
        plt.figure(figsize=(15, 10))
        for i in range(min(8, len(channels))):
            plt.subplot(2, 4, i+1)
            if hasattr(response_curves, 'shape') and len(response_curves.shape) >= 2:
                plt.plot(response_curves[:, i] if response_curves.shape[1] > i else [])
                plt.title(f'{channels[i]}')
                plt.xlabel('Spend')
                plt.ylabel('KPI Response')
        
        plt.tight_layout()
        path = f'{plot_dir}/{config_name}_03_response_curves.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_03_response_curves.png')
    except Exception as e:
        print(f"      ⚠ Response curves plot failed: {e}")
    
    # 4. Analysis summary
    try:
        plt.figure(figsize=(12, 8))
        
        summary_text = [
            f'MERIDIAN MMM ANALYSIS - {config_name}',
            '',
            'Model Results:',
            f'• Total KPI: {np.sum(mmm.input_data.kpi):,.0f} clicks',
            f'• Total Spend: €{np.sum(mmm.input_data.media_spend):,.0f}',
            ''
        ]
        
        if 'incremental_outcome' in analysis_results:
            if isinstance(analysis_results['incremental_outcome'], dict):
                total_inc = analysis_results['incremental_outcome'].get('total_incremental', 0)
                summary_text.append(f'• Total Incremental: {total_inc:,.0f} clicks')
        
        if 'roi' in analysis_results and isinstance(analysis_results['roi'], dict):
            mean_rois = analysis_results['roi'].get('mean_roi', [])
            if mean_rois:
                avg_roi = np.mean(mean_rois)
                summary_text.append(f'• Average ROI: {avg_roi:.2f}')
        
        if 'rhat' in analysis_results and isinstance(analysis_results['rhat'], dict):
            max_rhat = analysis_results['rhat'].get('max_rhat', 'N/A')
            summary_text.append(f'• Max R-hat: {max_rhat}')
        
        for i, text in enumerate(summary_text):
            plt.text(0.1, 0.9 - i*0.08, text, fontsize=11, transform=plt.gca().transAxes,
                    fontweight='bold' if i == 0 else 'normal')
        
        plt.axis('off')
        plt.title(f'Analysis Summary - {config_name}')
        
        path = f'{plot_dir}/{config_name}_04_analysis_summary.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_04_analysis_summary.png')
    except Exception as e:
        print(f"      ⚠ Summary plot failed: {e}")
    
    return plots_created

def test_prior_config(data, config):
    """Test a single prior configuration with full analysis"""
    print(f"\n🧪 Testing: {config['name']}")
    print(f"   ROI loc: {config['roi_loc']:.3f}, scale: {config['roi_scale']:.3f}")
    
    try:
        # Create custom prior
        roi_dist = tfp.distributions.LogNormal(
            loc=config['roi_loc'],
            scale=config['roi_scale'],
            name=constants.ROI_M
        )
        
        prior = prior_distribution.PriorDistribution(roi_m=roi_dist)
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        
        # Sample prior
        mmm.sample_prior(5)
        
        # Fit model
        mmm.sample_posterior(
            n_chains=4,
            n_adapt=1000,
            n_burnin=2000,
            n_keep=4000,
            seed=42
        )
        
        # Perform full analysis
        analysis_results = perform_full_analysis(mmm, config['name'])
        
        # Create analysis plots
        plots_created = create_analysis_plots(mmm, config['name'], analysis_results)
        
        result = {
            'name': config['name'],
            'status': 'SUCCESS',
            'model': mmm,
            'analysis': analysis_results,
            'plots': plots_created,
            'error': None
        }
        
        print(f"   ✓ SUCCESS - {len(plots_created)} plots created")
        return result
        
    except Exception as e:
        result = {
            'name': config['name'],
            'status': 'FAILED',
            'model': None,
            'analysis': None,
            'plots': [],
            'error': str(e)
        }
        print(f"   ✗ FAILED: {str(e)[:100]}...")
        return result

def save_results(results, data):
    """Save results and analysis"""
    print("\n📊 FULL ANALYSIS RESULTS")
    print("=" * 50)
    
    successful_configs = [r for r in results if r['status'] == 'SUCCESS']
    failed_configs = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✓ Successful: {len(successful_configs)}")
    print(f"✗ Failed: {len(failed_configs)}")
    
    if successful_configs:
        print("\n🏆 SUCCESSFUL CONFIGURATIONS:")
        for result in successful_configs:
            print(f"  - {result['name']}: {len(result['plots'])} plots created")
            
            # Save model and analysis
            model_path = f"/Users/aaronmeagher/Work/google_meridian/google/test/better_priors_2/{result['name']}_model.pkl"
            analysis_path = f"/Users/aaronmeagher/Work/google_meridian/google/test/better_priors_2/{result['name']}_analysis.pkl"
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                with open(analysis_path, 'wb') as f:
                    pickle.dump(result['analysis'], f)
                print(f"    ✓ Saved model and analysis files")
            except Exception as e:
                print(f"    ⚠ Save failed: {e}")
    
    if failed_configs:
        print("\n❌ FAILED CONFIGURATIONS:")
        for result in failed_configs:
            print(f"  - {result['name']}: {result['error'][:50]}...")

def main():
    print("🔬 MERIDIAN FULL ANALYSIS GRID SEARCH")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    data = load_data()
    print("✓ Data loaded successfully")
    
    # Get prior configurations
    configs = create_prior_configs()
    print(f"Testing {len(configs)} configurations with full MMM analysis...")
    
    # Test each configuration
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        result = test_prior_config(data, config)
        results.append(result)
    
    # Save and summarize results
    save_results(results, data)
    
    print("\n🎯 FULL MMM ANALYSIS COMPLETE!")
    print("Results saved in better_priors_2/ directory")
    print("Includes: incremental contribution, ROI, response curves, and diagnostics")

if __name__ == "__main__":
    main()