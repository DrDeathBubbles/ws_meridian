#!/usr/bin/env python3
"""
Meridian Ticket Sales Analysis
=============================
Run MMM analysis using ticket sales as KPI
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

def run_ticket_analysis():
    print("🎟️ MERIDIAN TICKET SALES ANALYSIS")
    print("=" * 50)
    
    # Create output directory
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load merged data
    coord_to_columns = load.CoordToColumns(
        time='date_id',
        geo='national_geo',
        kpi='ticket_sales',  # Using ticket sales as KPI
        media_spend=['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 
                    'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend'],
        media=['google_impressions', 'facebook_impressions', 'linkedin_impressions', 
              'reddit_impressions', 'bing_impressions', 'tiktok_impressions', 
              'twitter_impressions', 'instagram_impressions'],
    )
    
    loader = load.CsvDataLoader(
        csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/merged_marketing_tickets_fixed.csv',
        kpi_type='non_revenue',  # Using ticket sales directly
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
    
    data = loader.load()
    print("✓ Data loaded successfully")
    print(f"  Total ticket sales: {np.sum(data.kpi):,.0f}")
    print(f"  Total marketing spend: €{np.sum(data.media_spend):,.0f}")
    
    # Create model with data-driven priors
    # Conservative prior based on data analysis
    roi_loc = 1.044  # From data analysis
    roi_scale = 0.3
    
    roi_dist = tfp.distributions.LogNormal(
        loc=roi_loc,
        scale=roi_scale,
        name=constants.ROI_M
    )
    
    prior = prior_distribution.PriorDistribution(roi_m=roi_dist)
    model_spec = spec.ModelSpec(prior=prior)
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    print("✓ Model created with data-driven priors")
    
    # Sample prior
    mmm.sample_prior(5)
    print("✓ Prior sampling complete")
    
    # Fit model
    print("🔥 Fitting model (this will take several minutes)...")
    mmm.sample_posterior(
        n_chains=4,
        n_adapt=1000,
        n_burnin=2000,
        n_keep=4000,
        seed=42
    )
    print("✓ Model fitted successfully!")
    
    # Save the fitted model
    model_path = f"{output_dir}/ticket_sales_model.pkl"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(mmm, f)
        print(f"✓ Model saved: {model_path}")
    except Exception as save_error:
        print(f"⚠ Model save failed: {save_error}")
    
    # Analyze results
    print("\n📊 ANALYZING RESULTS...")
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # 1. Incremental contribution
    try:
        incremental = analyzer_obj.incremental_outcome(use_kpi=True)
        total_incremental = np.sum(incremental)
        channel_contribution = [np.sum(incremental[:,:,i]) for i in range(incremental.shape[2])]
        
        print(f"✓ Incremental analysis complete")
        print(f"  Total incremental tickets: {total_incremental:,.0f}")
        print(f"  Percent of total: {total_incremental/np.sum(data.kpi)*100:.1f}%")
        
        # Plot incremental contribution
        channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
        
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(channel_contribution)], channel_contribution)
        plt.title('Incremental Ticket Sales by Channel')
        plt.xlabel('Channel')
        plt.ylabel('Incremental Tickets')
        plt.xticks(rotation=45)
        
        path = f"{output_dir}/incremental_tickets_by_channel.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: incremental_tickets_by_channel.png")
        
    except Exception as e:
        print(f"⚠ Incremental analysis failed: {e}")
    
    # 2. ROI analysis using built-in method
    try:
        # Define channels list here to ensure it's available
        channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
        
        # Get ROI using built-in method with correct averaging
        roi_results = analyzer_obj.roi(use_kpi=True)
        print(f"   ROI tensor shape: {roi_results.shape}")
        
        # Average across chains (axis 0) and samples (axis 1) to get one value per channel
        mean_roi = np.mean(roi_results, axis=(0, 1))
        print(f"   Mean ROI shape: {mean_roi.shape}")
        
        # Plot ROI
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(mean_roi)], mean_roi)
        plt.title('ROI by Channel (Tickets per €)')
        plt.xlabel('Channel')
        plt.ylabel('ROI (Tickets/€)')
        plt.xticks(rotation=45)
        
        path = f"{output_dir}/roi_by_channel.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: roi_by_channel.png")
        
        # Print ROI values
        print("\n📈 ROI ANALYSIS (Tickets per €):")
        for i, channel in enumerate(channels[:len(mean_roi)]):
            print(f"  {channel}: {mean_roi[i]:.3f}")
            
    except Exception as e:
        print(f"⚠ ROI analysis failed: {e}")
        
        # Fall back to manual calculation if built-in method fails
        try:
            print("   Falling back to manual ROI calculation...")
            incremental = analyzer_obj.incremental_outcome(use_kpi=True)
            total_spend = np.sum(data.media_spend, axis=(0,1))  # Sum spend by channel
            
            # Calculate total contribution by channel
            channel_contribution = [np.sum(incremental[:,:,i]) for i in range(incremental.shape[2])]
            
            # Calculate ROI as contribution / spend
            roi_values = []
            for i in range(len(channel_contribution)):
                if total_spend[i] > 0:
                    roi = channel_contribution[i] / total_spend[i]
                else:
                    roi = 0
                roi_values.append(roi)
            
            # Plot ROI
            plt.figure(figsize=(12, 6))
            plt.bar(channels[:len(roi_values)], roi_values)
            plt.title('ROI by Channel (Tickets per €) - Manual Calculation')
            plt.xlabel('Channel')
            plt.ylabel('ROI (Tickets/€)')
            plt.xticks(rotation=45)
            
            path = f"{output_dir}/roi_by_channel_manual.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: roi_by_channel_manual.png")
            
            # Print ROI values
            print("\n📈 MANUAL ROI ANALYSIS (Tickets per €):")
            for i, channel in enumerate(channels[:len(roi_values)]):
                print(f"  {channel}: {roi_values[i]:.3f}")
                
        except Exception as inner_e:
            print(f"⚠ Manual ROI calculation failed: {inner_e}")
        
    except Exception as e:
        print(f"⚠ ROI analysis failed: {e}")
    
    # 3. Response curves - with 100 points and consistent scales
    try:
        # Define channels list here to ensure it's available
        channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
        
        # Create a figure for response curves
        fig, axs = plt.subplots(2, 4, figsize=(15, 10), sharex=False, sharey=False)
        axs = axs.flatten()
        
        # Get total spend for scaling
        total_spend = np.sum(data.media_spend, axis=(0,1))
        
        # Get incremental contribution for reference
        incremental = analyzer_obj.incremental_outcome(use_kpi=True)
        channel_contribution = [float(np.sum(incremental[:,:,i])) for i in range(incremental.shape[2])]
        
        # Find max values for consistent scaling
        max_spend = float(np.max(total_spend)) * 2.0  # 200% of max spend
        max_contribution = float(np.max(channel_contribution)) * 1.2  # 120% of max contribution
        
        # Create response curves for each channel
        for i in range(min(8, len(channels))):
            ax = axs[i]
            
            # Create spend levels from 0% to 200% of current spend with 100 points
            if total_spend[i] > 0:
                current_spend = float(total_spend[i])
                current_contrib = float(channel_contribution[i])
                
                if current_contrib > 0:
                    # Create 100 points from 0 to 200% of current spend
                    x = np.linspace(0, current_spend * 2.0, 100)
                    
                    # Simple diminishing returns curve (square root model)
                    scale_factor = current_contrib / np.sqrt(current_spend)
                    y = scale_factor * np.sqrt(x)
                    
                    # Plot the curve
                    ax.plot(x, y)
                    ax.axvline(x=current_spend, color='r', linestyle='--', alpha=0.3)
                    ax.axhline(y=current_contrib, color='r', linestyle='--', alpha=0.3)
                    ax.scatter([current_spend], [current_contrib], color='red', zorder=5)
                    
                    # Set consistent scales
                    ax.set_xlim([0, max_spend])
                    ax.set_ylim([0, max_contribution])
                    
                    # Add labels
                    ax.set_title(channels[i])
                    ax.set_xlabel('Spend (€)')
                    ax.set_ylabel('Ticket Sales')
                else:
                    ax.text(0.5, 0.5, 'No contribution', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim([0, max_spend])
                    ax.set_ylim([0, max_contribution])
            else:
                ax.text(0.5, 0.5, 'No spend data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim([0, max_spend])
                ax.set_ylim([0, max_contribution])
        
        # Add a title for the entire figure
        fig.suptitle('Response Curves: Ticket Sales vs. Marketing Spend', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        
        path = f"{output_dir}/response_curves.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: response_curves.png")
        
        # Create log-log version of response curves
        fig, axs = plt.subplots(2, 4, figsize=(15, 10), sharex=False, sharey=False)
        axs = axs.flatten()
        
        # Create response curves for each channel with log-log scale
        for i in range(min(8, len(channels))):
            ax = axs[i]
            
            # Create spend levels from 0.1% to 200% of current spend with 100 points
            if total_spend[i] > 0:
                current_spend = float(total_spend[i])
                current_contrib = float(channel_contribution[i])
                
                if current_contrib > 0:
                    # Create 100 points from 0.1% to 200% of current spend (avoid zero for log scale)
                    x = np.linspace(current_spend * 0.001, current_spend * 2.0, 100)
                    
                    # Simple diminishing returns curve (square root model)
                    scale_factor = current_contrib / np.sqrt(current_spend)
                    y = scale_factor * np.sqrt(x)
                    
                    # Plot the curve with log-log scale
                    ax.loglog(x, y)
                    ax.axvline(x=current_spend, color='r', linestyle='--', alpha=0.3)
                    ax.axhline(y=current_contrib, color='r', linestyle='--', alpha=0.3)
                    ax.scatter([current_spend], [current_contrib], color='red', zorder=5)
                    
                    # Add labels
                    ax.set_title(channels[i])
                    ax.set_xlabel('Spend (€) - Log Scale')
                    ax.set_ylabel('Ticket Sales - Log Scale')
                    
                    # Add grid for log scale
                    ax.grid(True, which="both", ls="-", alpha=0.2)
                else:
                    ax.text(0.5, 0.5, 'No contribution', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'No spend data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        # Add a title for the log-log figure
        fig.suptitle('Response Curves (Log-Log Scale): Ticket Sales vs. Marketing Spend', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        
        path = f"{output_dir}/response_curves_loglog.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: response_curves_loglog.png")
        
    except Exception as e:
        print(f"⚠ Response curves failed: {e}")
    
    # 4. Baseline vs incremental
    try:
        baseline = analyzer_obj.baseline_summary_metrics()
        
        # Create time series plot
        plt.figure(figsize=(15, 8))
        
        # Get time series data
        kpi_time = np.mean(data.kpi, axis=0)
        
        # Plot
        plt.plot(kpi_time, label='Total Ticket Sales', linewidth=2)
        
        # Add trend line
        x = np.arange(len(kpi_time))
        z = np.polyfit(x, kpi_time, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), label='Trend', linestyle='--', alpha=0.7)
        
        plt.title('Ticket Sales Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Ticket Sales')
        plt.legend()
        
        path = f"{output_dir}/ticket_sales_over_time.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: ticket_sales_over_time.png")
        
    except Exception as e:
        print(f"⚠ Baseline analysis failed: {e}")
    
    print("\n🎯 TICKET SALES ANALYSIS COMPLETE!")
    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    run_ticket_analysis()