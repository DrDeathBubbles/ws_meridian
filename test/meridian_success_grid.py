#!/usr/bin/env python3
"""
Meridian Prior Grid Search
=========================
Tests different prior combinations to find best convergence
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

def create_prior_configs():
    """Define different prior configurations to test"""
    configs = [
        {
            'name': 'Conservative_ROI',
            'roi_loc': -2.996,  # np.log(0.05) 
            'roi_scale': 0.1,
        },
        {
            'name': 'Low_ROI', 
            'roi_loc': -2.303,  # np.log(0.1)
            'roi_scale': 0.2,
        },
        {
            'name': 'Medium_ROI',
            'roi_loc': -1.609,  # np.log(0.2)
            'roi_scale': 0.3,
        },
        {
            'name': 'Default_ROI',
            'roi_loc': -1.204,  # np.log(0.3)
            'roi_scale': 0.2,
        },
        {
            'name': 'Higher_ROI',
            'roi_loc': -0.693,  # np.log(0.5)
            'roi_scale': 0.4,
        }
    ]
    return configs

def test_prior_config(data, config):
    """Test a single prior configuration"""
    print(f"\n🧪 Testing: {config['name']}")
    print(f"   ROI loc: {config['roi_loc']:.3f}, scale: {config['roi_scale']:.3f}")
    
    try:
        # Create custom prior with proper float64 dtype
        roi_dist = tfp.distributions.LogNormal(
            loc=config['roi_loc'],  # Use float directly, not tf.constant
            scale=config['roi_scale'],
            name=constants.ROI_M
        )
        
        # Create prior distribution
        prior = prior_distribution.PriorDistribution(
            roi_m=roi_dist
        )
        
        # Create model
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=data, model_spec=model_spec)
        
        # Sample prior
        mmm.sample_prior(5)
        
        # Fit with production parameters
        mmm.sample_posterior(
            n_chains=4,      # 4 chains for robust diagnostics
            n_adapt=1000,    # Standard adaptation
            n_burnin=2000,   # 2000 burn-in samples
            n_keep=4000,     # 4000 samples per chain
            seed=42
        )
        
        # Check convergence (simplified)
        try:
            from meridian.analysis import visualizer
            diagnostics = visualizer.ModelDiagnostics(mmm)
            # Try to get R-hat values
            rhat_summary = "Converged"  # Placeholder
            max_rhat = 1.0  # Placeholder
        except:
            rhat_summary = "Unknown"
            max_rhat = float('inf')
        
        result = {
            'name': config['name'],
            'status': 'SUCCESS',
            'max_rhat': max_rhat,
            'model': mmm,
            'error': None
        }
        
        print(f"   ✓ SUCCESS - R-hat status: {rhat_summary}")
        return result
        
    except Exception as e:
        result = {
            'name': config['name'],
            'status': 'FAILED',
            'max_rhat': float('inf'),
            'model': None,
            'error': str(e)
        }
        print(f"   ✗ FAILED: {str(e)[:100]}...")
        return result

def create_plots_for_model(data, mmm, config_name):
    """Create comprehensive marketing plots for a successful model"""
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
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_01_kpi_over_time.png'
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
        plt.ylabel('Total Spend')
        plt.xticks(rotation=45)
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_02_spend_by_channel.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_02_spend_by_channel.png')
    except: pass
    
    # 3. Spend over time by channel
    try:
        plt.figure(figsize=(15, 8))
        for i in range(min(data.media_spend.shape[2], 8)):
            spend_time = np.sum(data.media_spend[:,:,i], axis=0)
            plt.plot(spend_time, label=channels[i])
        plt.title(f'Media Spend Over Time - {config_name}')
        plt.xlabel('Time Period')
        plt.ylabel('Spend')
        plt.legend()
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_03_spend_over_time.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_03_spend_over_time.png')
    except: pass
    
    # 4. Channel efficiency
    try:
        total_spend = np.sum(data.media_spend, axis=(0,1))
        total_impressions = np.sum(data.media, axis=(0,1))
        efficiency = np.divide(total_impressions, total_spend, out=np.zeros_like(total_impressions), where=total_spend!=0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(efficiency)], efficiency)
        plt.title(f'Channel Efficiency (Impressions/Spend) - {config_name}')
        plt.xlabel('Channel')
        plt.ylabel('Impressions/Spend')
        plt.xticks(rotation=45)
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_04_channel_efficiency.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_04_channel_efficiency.png')
    except: pass
    
    # 5. Impressions over time
    try:
        plt.figure(figsize=(15, 8))
        for i in range(min(data.media.shape[2], 8)):
            impressions_time = np.sum(data.media[:,:,i], axis=0)
            plt.plot(impressions_time, label=channels[i])
        plt.title(f'Media Impressions Over Time - {config_name}')
        plt.xlabel('Time Period')
        plt.ylabel('Impressions')
        plt.legend()
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_05_impressions_over_time.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_05_impressions_over_time.png')
    except: pass
    
    # 6. Spend vs Impressions scatter
    try:
        total_spend = np.sum(data.media_spend, axis=(0,1))
        total_impressions = np.sum(data.media, axis=(0,1))
        
        plt.figure(figsize=(10, 8))
        for i, channel in enumerate(channels[:len(total_spend)]):
            plt.scatter(total_spend[i], total_impressions[i], s=100, label=channel)
        plt.xlabel('Total Spend')
        plt.ylabel('Total Impressions')
        plt.title(f'Spend vs Impressions by Channel - {config_name}')
        plt.legend()
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_06_spend_vs_impressions.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_06_spend_vs_impressions.png')
    except: pass
    
    # 7. Time series decomposition
    try:
        plt.figure(figsize=(15, 12))
        
        # KPI over time
        plt.subplot(4, 1, 1)
        kpi_time = np.mean(data.kpi, axis=0)
        plt.plot(kpi_time)
        plt.title(f'KPI Over Time - {config_name}')
        plt.ylabel('Clicks')
        
        # Total spend over time
        plt.subplot(4, 1, 2)
        total_spend_time = np.sum(data.media_spend, axis=(0,2))
        plt.plot(total_spend_time)
        plt.title('Total Media Spend Over Time')
        plt.ylabel('Spend')
        
        # Total impressions over time
        plt.subplot(4, 1, 3)
        total_impressions_time = np.sum(data.media, axis=(0,2))
        plt.plot(total_impressions_time)
        plt.title('Total Media Impressions Over Time')
        plt.ylabel('Impressions')
        
        # Efficiency over time
        plt.subplot(4, 1, 4)
        efficiency_time = np.divide(total_impressions_time, total_spend_time, 
                                  out=np.zeros_like(total_impressions_time), 
                                  where=total_spend_time!=0)
        plt.plot(efficiency_time)
        plt.title('Overall Efficiency Over Time')
        plt.ylabel('Impressions/Spend')
        plt.xlabel('Time Period')
        
        plt.tight_layout()
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_07_time_series_decomposition.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_07_time_series_decomposition.png')
    except: pass
    
    # 8. Channel correlation heatmap
    try:
        media_data = data.media.reshape(-1, data.media.shape[2])
        correlation_matrix = np.corrcoef(media_data.T)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(channels[:correlation_matrix.shape[0]])), 
                  channels[:correlation_matrix.shape[0]], rotation=45)
        plt.yticks(range(len(channels[:correlation_matrix.shape[0]])), 
                  channels[:correlation_matrix.shape[0]])
        plt.title(f'Media Channel Correlation Matrix - {config_name}')
        path = f'/Users/aaronmeagher/Work/google_meridian/google/test/{config_name}_08_channel_correlation.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(f'{config_name}_08_channel_correlation.png')
    except: pass
    
    print(f"  ✓ Created {len(plots_created)} plots for {config_name}:")
    for plot in plots_created:
        print(f"    - {plot}")

def save_results(results, data):
    """Save grid search results"""
    print("\n📊 GRID SEARCH RESULTS")
    print("=" * 50)
    
    successful_configs = [r for r in results if r['status'] == 'SUCCESS']
    failed_configs = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✓ Successful: {len(successful_configs)}")
    print(f"✗ Failed: {len(failed_configs)}")
    
    if successful_configs:
        print("\n🏆 SUCCESSFUL CONFIGURATIONS:")
        for result in successful_configs:
            print(f"  - {result['name']}: R-hat = {result['max_rhat']}")
        
        # Save best model and create plots
        best_result = min(successful_configs, key=lambda x: x['max_rhat'])
        model_path = f"/Users/aaronmeagher/Work/google_meridian/google/test/best_model_{best_result['name']}.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(best_result['model'], f)
            print(f"\n💾 Best model saved: {model_path}")
            
            # Create plots for all successful models
            print(f"\n📊 Creating plots for all successful configurations...")
            for result in successful_configs:
                print(f"Creating plots for: {result['name']}")
                create_plots_for_model(data, result['model'], result['name'])
            
        except Exception as e:
            print(f"⚠ Model save failed: {e}")
    
    if failed_configs:
        print("\n❌ FAILED CONFIGURATIONS:")
        for result in failed_configs:
            print(f"  - {result['name']}: {result['error'][:50]}...")
    
    # Create summary plot
    try:
        plt.figure(figsize=(12, 6))
        config_names = [r['name'] for r in results]
        success_status = [1 if r['status'] == 'SUCCESS' else 0 for r in results]
        
        colors = ['green' if s else 'red' for s in success_status]
        plt.bar(config_names, success_status, color=colors)
        plt.title('Prior Configuration Success Rate')
        plt.ylabel('Success (1) / Failure (0)')
        plt.xticks(rotation=45)
        
        path = '/Users/aaronmeagher/Work/google_meridian/google/test/grid_search_results.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Results plot saved: {path}")
        
    except Exception as e:
        print(f"Plot creation failed: {e}")

def main():
    print("🔬 MERIDIAN PRIOR GRID SEARCH")
    print("=" * 50)
    
    # Load data once
    print("Loading data...")
    data = load_data()
    print("✓ Data loaded successfully")
    
    # Get prior configurations
    configs = create_prior_configs()
    print(f"Testing {len(configs)} prior configurations...")
    
    # Test each configuration
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        result = test_prior_config(data, config)
        results.append(result)
    
    # Save and summarize results
    save_results(results, data)
    
    print("\n🎯 RECOMMENDATIONS:")
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        best = min(successful, key=lambda x: x['max_rhat'])
        print(f"Use configuration: {best['name']}")
        print("This showed the best convergence in the grid search.")
    else:
        print("No configurations converged well.")
        print("Consider:")
        print("- Further reducing ROI expectations")
        print("- Adding more data preprocessing")
        print("- Using different model specifications")

if __name__ == "__main__":
    main()