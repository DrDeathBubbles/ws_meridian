#!/usr/bin/env python3
"""
Meridian Data-Driven Prior Testing
=================================
Tests conservative and moderate priors based on actual data
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

def analyze_immediately_after_fitting(mmm, config_name):
    """Analyze model immediately after fitting to capture insights"""
    analysis = {}
    
    try:
        # Try to access analyzer
        from meridian.analysis import analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        
        # Check available methods
        methods = [m for m in dir(analyzer_obj) if not m.startswith('_')]
        analysis['available_methods'] = methods[:10]
        
        # Try different analysis approaches
        for method_name in ['get_posterior_metrics', 'get_summary', 'get_results']:
            if hasattr(analyzer_obj, method_name):
                try:
                    result = getattr(analyzer_obj, method_name)()
                    analysis[method_name] = f"Available: {type(result)}"
                except Exception as e:
                    analysis[method_name] = f"Failed: {str(e)[:50]}"
        
        # Model structure analysis
        analysis['model_structure'] = {
            'kpi_shape': mmm.input_data.kpi.shape,
            'media_shape': mmm.input_data.media.shape,
            'total_kpi': float(np.sum(mmm.input_data.kpi)),
            'total_spend': float(np.sum(mmm.input_data.media_spend))
        }
        
        # Check for posterior samples in different locations
        sample_locations = ['posterior_samples', '_posterior_samples', 'samples', 'mcmc_samples']
        for loc in sample_locations:
            if hasattr(mmm, loc):
                samples = getattr(mmm, loc)
                if samples is not None:
                    analysis[f'samples_{loc}'] = f"Found: {type(samples)}"
                    
                    # Try to extract sample info
                    if hasattr(samples, 'posterior'):
                        analysis['posterior_vars'] = list(samples.posterior.keys())[:5]
                    break
        
        print(f"   ✓ Analysis captured for {config_name}")
        
    except Exception as e:
        analysis['error'] = str(e)
        print(f"   ⚠ Analysis failed: {str(e)[:50]}")
    
    return analysis

def create_prior_configs():
    """Data-driven prior configurations"""
    configs = [
        {
            'name': 'Conservative',
            'roi_loc': 1.044,   # From data analysis
            'roi_scale': 0.3,
        },
        {
            'name': 'Moderate',
            'roi_loc': 1.737,   # From data analysis
            'roi_scale': 0.5,
        }
    ]
    return configs

def test_prior_config(data, config):
    """Test a single prior configuration"""
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
        
        # Immediate analysis after fitting
        analysis_results = analyze_immediately_after_fitting(mmm, config['name'])
        
        result = {
            'name': config['name'],
            'status': 'SUCCESS',
            'model': mmm,
            'analysis': analysis_results,
            'error': None
        }
        
        print(f"   ✓ SUCCESS")
        return result
        
    except Exception as e:
        result = {
            'name': config['name'],
            'status': 'FAILED',
            'model': None,
            'error': str(e)
        }
        print(f"   ✗ FAILED: {str(e)[:100]}...")
        return result

def create_plots_for_model(data, mmm, config_name):
    """Create comprehensive marketing plots"""
    # Create subdirectory
    plot_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors'
    os.makedirs(plot_dir, exist_ok=True)
    
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    plots_created = []
    
    # 1. KPI over time
    try:
        plt.figure(figsize=(12, 6))
        kpi_time = np.mean(data.kpi, axis=0)
        plt.plot(kpi_time)
        plt.title(f'KPI Over Time - {config_name}')
        plt.xlabel('Time Period')
        plt.ylabel('Clicks')
        path = f'{plot_dir}/{config_name}_01_kpi_over_time.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_01_kpi_over_time.png')
    except: pass
    
    # 2. Total spend by channel
    try:
        total_spend = np.sum(data.media_spend, axis=(0,1))
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(total_spend)], total_spend)
        plt.title(f'Total Media Spend by Channel - {config_name}')
        plt.xlabel('Channel')
        plt.ylabel('Total Spend (€)')
        plt.xticks(rotation=45)
        path = f'{plot_dir}/{config_name}_02_spend_by_channel.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_02_spend_by_channel.png')
    except: pass
    
    # 3. Channel efficiency
    try:
        total_spend = np.sum(data.media_spend, axis=(0,1))
        total_impressions = np.sum(data.media, axis=(0,1))
        efficiency = np.divide(total_impressions, total_spend, out=np.zeros_like(total_impressions), where=total_spend!=0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(efficiency)], efficiency)
        plt.title(f'Channel Efficiency (Impressions/Spend) - {config_name}')
        plt.xlabel('Channel')
        plt.ylabel('Impressions per €')
        plt.xticks(rotation=45)
        path = f'{plot_dir}/{config_name}_03_channel_efficiency.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_03_channel_efficiency.png')
    except: pass
    
    # 4. Spend over time
    try:
        plt.figure(figsize=(15, 8))
        for i in range(min(data.media_spend.shape[2], 8)):
            spend_time = np.sum(data.media_spend[:,:,i], axis=0)
            if np.sum(spend_time) > 0:  # Only plot channels with spend
                plt.plot(spend_time, label=channels[i])
        plt.title(f'Media Spend Over Time - {config_name}')
        plt.xlabel('Time Period')
        plt.ylabel('Spend (€)')
        plt.legend()
        path = f'{plot_dir}/{config_name}_04_spend_over_time.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_04_spend_over_time.png')
    except: pass
    
    # 5. Impressions over time
    try:
        plt.figure(figsize=(15, 8))
        for i in range(min(data.media.shape[2], 8)):
            impressions_time = np.sum(data.media[:,:,i], axis=0)
            if np.sum(impressions_time) > 0:  # Only plot channels with impressions
                plt.plot(impressions_time, label=channels[i])
        plt.title(f'Media Impressions Over Time - {config_name}')
        plt.xlabel('Time Period')
        plt.ylabel('Impressions')
        plt.legend()
        path = f'{plot_dir}/{config_name}_05_impressions_over_time.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_05_impressions_over_time.png')
    except: pass
    
    print(f"  ✓ Created {len(plots_created)} plots for {config_name}")
    return plots_created

def save_results(results, data):
    """Save results and create plots"""
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    successful_configs = [r for r in results if r['status'] == 'SUCCESS']
    failed_configs = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✓ Successful: {len(successful_configs)}")
    print(f"✗ Failed: {len(failed_configs)}")
    
    if successful_configs:
        print("\n🏆 SUCCESSFUL CONFIGURATIONS:")
        for result in successful_configs:
            print(f"  - {result['name']}")
        
        # Create plots for all successful models
        print(f"\n📊 Creating plots in better_priors/ directory...")
        for result in successful_configs:
            print(f"Creating plots for: {result['name']}")
            create_plots_for_model(data, result['model'], result['name'])
            
            # Save model
            model_path = f"/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/{result['name']}_model.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                print(f"  ✓ Model saved: {result['name']}_model.pkl")
            except Exception as e:
                print(f"  ⚠ Model save failed: {e}")
            
            # Save analysis results
            if 'analysis' in result:
                analysis_path = f"/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/{result['name']}_analysis.pkl"
                try:
                    with open(analysis_path, 'wb') as f:
                        pickle.dump(result['analysis'], f)
                    print(f"  ✓ Analysis saved: {result['name']}_analysis.pkl")
                    
                    # Print key insights
                    analysis = result['analysis']
                    if 'model_structure' in analysis:
                        struct = analysis['model_structure']
                        print(f"    Total KPI: {struct['total_kpi']:,.0f}")
                        print(f"    Total Spend: €{struct['total_spend']:,.0f}")
                    
                except Exception as e:
                    print(f"  ⚠ Analysis save failed: {e}")
    
    if failed_configs:
        print("\n❌ FAILED CONFIGURATIONS:")
        for result in failed_configs:
            print(f"  - {result['name']}: {result['error'][:50]}...")

def main():
    print("🔬 MERIDIAN DATA-DRIVEN PRIOR TESTING")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    data = load_data()
    print("✓ Data loaded successfully")
    
    # Get prior configurations
    configs = create_prior_configs()
    print(f"Testing {len(configs)} data-driven prior configurations...")
    
    # Test each configuration
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        result = test_prior_config(data, config)
        results.append(result)
    
    # Save and summarize results
    save_results(results, data)
    
    print("\n🎯 SUMMARY:")
    print("Data-driven priors tested based on your Web Summit data")
    print("Plots saved in better_priors/ subdirectory")
    print("Models saved as pickle files for reuse")

if __name__ == "__main__":
    main()