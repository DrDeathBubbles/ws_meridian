#!/usr/bin/env python3
"""
Marketing Channel Attribution
===========================
Extracts and visualizes the contribution of each marketing channel from the Meridian model.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Set paths
MODEL_PATH = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model/improved_model.pkl'
OUTPUT_DIR = '/Users/aaronmeagher/Work/google_meridian/google/test/improved_model'

def main():
    print("Loading model...")
    
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print("Extracting channel attribution...")
    
    # Create analyzer object
    from meridian.analysis import analyzer
    analyzer_obj = analyzer.Analyzer(model)
    
    # Get channel names
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Get incremental contribution by channel
    try:
        print("Calculating incremental contribution...")
        incremental = analyzer_obj.incremental_outcome(use_kpi=True)
        print(f"Incremental shape: {incremental.shape}")
        
        # Sum contribution by channel
        channel_contribution = [np.sum(incremental[:,:,i]) for i in range(incremental.shape[2])]
        
        # Create bar chart of total contribution by channel
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(channel_contribution)], channel_contribution)
        plt.title('Total Contribution by Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Contribution', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'channel_contribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Channel contribution plot saved to: {output_path}")
        
        # Print contribution by channel
        print("\nChannel Contribution:")
        for i, channel in enumerate(channels[:len(channel_contribution)]):
            print(f"  {channel}: {channel_contribution[i]:.2f}")
        
        # Calculate ROI by channel
        print("\nCalculating ROI by channel...")
        total_spend = np.sum(model.input_data.media_spend.values, axis=(0,1))  # Sum spend by channel
        
        # Calculate ROI as contribution / spend
        roi_values = []
        for i in range(len(channel_contribution)):
            if total_spend[i] > 0:
                roi = channel_contribution[i] / total_spend[i]
            else:
                roi = 0
            roi_values.append(roi)
        
        # Create bar chart of ROI by channel
        plt.figure(figsize=(12, 6))
        plt.bar(channels[:len(roi_values)], roi_values)
        plt.title('ROI by Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('ROI (Contribution/Spend)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'channel_roi.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Channel ROI plot saved to: {output_path}")
        
        # Print ROI by channel
        print("\nChannel ROI:")
        for i, channel in enumerate(channels[:len(roi_values)]):
            print(f"  {channel}: {roi_values[i]:.4f}")
        
        # Create time series plot of channel contribution
        print("\nCreating time series plot of channel contribution...")
        
        # Sum across geos
        if len(incremental.shape) > 2:
            incremental_by_time = np.sum(incremental, axis=0)  # Sum across geos
        else:
            incremental_by_time = incremental
        
        # Create stacked area chart
        plt.figure(figsize=(14, 8))
        x = np.arange(incremental_by_time.shape[0])
        
        # Create stacked area chart
        plt.stackplot(x, 
                     [incremental_by_time[:, i] for i in range(incremental_by_time.shape[1])],
                     labels=channels[:incremental_by_time.shape[1]],
                     alpha=0.7)
        
        plt.title('Channel Contribution Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Contribution', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Save plot
        output_path = os.path.join(OUTPUT_DIR, 'channel_contribution_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Channel contribution time series plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error calculating channel attribution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()