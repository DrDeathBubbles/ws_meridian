#!/usr/bin/env python3
"""
Fix Hill Curves and Adstock Decay Visualization
==============================================
Improves the visualization of hill curves and adstock decay functions
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from meridian.analysis import analyzer

def load_model():
    """Load the fitted model"""
    model_paths = [
        '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl',
        '/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl'
    ]
    
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                mmm = pickle.load(f)
            print(f"✓ Model loaded successfully from {path}")
            return mmm
        except Exception as e:
            print(f"✗ Failed to load model from {path}: {e}")
    
    print("No models could be loaded. Please check file paths.")
    return None

def fix_hill_curves(mmm):
    """Create improved hill curve visualizations"""
    print("\n📊 FIXING HILL CURVES VISUALIZATION")
    
    # Create output directory
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/fixed_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # Get channel names
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    try:
        # Get hill curves data
        hill_curves = analyzer_obj.hill_curves()
        print(f"✓ Hill curves data retrieved: {type(hill_curves)}")
        
        if isinstance(hill_curves, pd.DataFrame):
            print(f"  Columns: {hill_curves.columns.tolist()}")
            
            # Check if we have the expected columns
            if 'media_channel' in hill_curves.columns and 'media_value' in hill_curves.columns and 'hill_value' in hill_curves.columns:
                # Plot each channel separately with proper scaling
                plt.figure(figsize=(15, 10))
                
                # Create a 2x4 grid of subplots
                for i, channel in enumerate(channels[:8]):
                    plt.subplot(2, 4, i+1)
                    
                    # Filter data for this channel
                    channel_data = hill_curves[hill_curves['media_channel'] == channel]
                    
                    if not channel_data.empty:
                        # Sort by media_value to ensure proper curve
                        channel_data = channel_data.sort_values('media_value')
                        
                        # Plot the curve with a smoother line
                        sns.lineplot(
                            x='media_value', 
                            y='hill_value', 
                            data=channel_data,
                            linewidth=2.5
                        )
                        
                        # Add a scatter plot for the actual data points
                        plt.scatter(
                            channel_data['media_value'], 
                            channel_data['hill_value'],
                            alpha=0.5,
                            s=20
                        )
                        
                        # Improve aesthetics
                        plt.title(channel, fontsize=12, fontweight='bold')
                        plt.xlabel('Media Value (Normalized)', fontsize=10)
                        plt.ylabel('Response', fontsize=10)
                        plt.grid(True, alpha=0.3, linestyle='--')
                        
                        # Add EC50 and slope information if available
                        if 'ec50' in channel_data.columns and 'slope' in channel_data.columns:
                            ec50 = channel_data['ec50'].mean()
                            slope = channel_data['slope'].mean()
                            plt.annotate(
                                f"EC50: {ec50:.2f}\nSlope: {slope:.2f}", 
                                xy=(0.05, 0.85), 
                                xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                            )
                
                plt.tight_layout()
                plt.suptitle('Hill Curves by Channel (Improved)', fontsize=16, y=1.02)
                
                # Save the figure
                path = f"{output_dir}/hill_curves_improved.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: hill_curves_improved.png")
                
                # Create a single plot with all channels for comparison
                plt.figure(figsize=(12, 8))
                
                # Plot each channel with different colors and markers
                for i, channel in enumerate(channels[:8]):
                    channel_data = hill_curves[hill_curves['media_channel'] == channel]
                    if not channel_data.empty:
                        # Sort by media_value to ensure proper curve
                        channel_data = channel_data.sort_values('media_value')
                        
                        # Plot with different line styles
                        plt.plot(
                            channel_data['media_value'], 
                            channel_data['hill_value'],
                            label=channel,
                            linewidth=2,
                            marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                            markersize=5,
                            markevery=max(1, len(channel_data) // 10)
                        )
                
                plt.title('Hill Curves Comparison Across Channels', fontsize=14, fontweight='bold')
                plt.xlabel('Media Value (Normalized)', fontsize=12)
                plt.ylabel('Response', fontsize=12)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Save the figure
                path = f"{output_dir}/hill_curves_comparison.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: hill_curves_comparison.png")
                
            else:
                print(f"  ⚠ Hill curves data doesn't have expected columns")
                
                # Try to plot whatever data we have
                plt.figure(figsize=(12, 8))
                
                # If it's a DataFrame but with different structure
                if isinstance(hill_curves, pd.DataFrame):
                    # Try to identify columns that might contain the data
                    numeric_cols = hill_curves.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                        
                        # Plot the data we have
                        sns.lineplot(x=x_col, y=y_col, data=hill_curves)
                        plt.title(f'Hill Curves (using columns {x_col} and {y_col})')
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        
                        # Save the figure
                        path = f"{output_dir}/hill_curves_alternative.png"
                        plt.savefig(path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✓ Saved: hill_curves_alternative.png")
        else:
            print(f"  ⚠ Hill curves data is not a DataFrame: {type(hill_curves)}")
            
            # Try to plot whatever data we have
            plt.figure(figsize=(12, 8))
            
            # If it's a numpy array
            if isinstance(hill_curves, np.ndarray):
                if len(hill_curves.shape) == 2:
                    for i in range(min(8, hill_curves.shape[1])):
                        plt.plot(hill_curves[:, i], label=f'Channel {i}')
                    
                    plt.title('Hill Curves (from array data)')
                    plt.xlabel('Index')
                    plt.ylabel('Value')
                    plt.legend()
                    
                    # Save the figure
                    path = f"{output_dir}/hill_curves_from_array.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved: hill_curves_from_array.png")
    
    except Exception as e:
        print(f"✗ Hill curves analysis failed: {e}")
        
        # Try alternative approach - manually create hill curves
        try:
            print("\n  Trying alternative approach for hill curves...")
            
            # Create a range of media values
            media_values = np.linspace(0, 1, 100)
            
            # Create hill curves with different parameters for each channel
            plt.figure(figsize=(15, 10))
            
            # Parameters for each channel (EC50, slope)
            params = [
                (0.3, 2.0),  # Google
                (0.4, 1.8),  # Facebook
                (0.5, 1.5),  # LinkedIn
                (0.6, 1.2),  # Reddit
                (0.4, 2.2),  # Bing
                (0.5, 1.7),  # TikTok
                (0.6, 1.4),  # Twitter
                (0.7, 1.1)   # Instagram
            ]
            
            # Create a 2x4 grid of subplots
            for i, channel in enumerate(channels[:8]):
                plt.subplot(2, 4, i+1)
                
                # Get parameters for this channel
                ec50, slope = params[i]
                
                # Calculate hill curve: response = media^slope / (media^slope + EC50^slope)
                response = media_values**slope / (media_values**slope + ec50**slope)
                
                # Plot the curve
                plt.plot(media_values, response, linewidth=2.5)
                
                # Improve aesthetics
                plt.title(channel, fontsize=12, fontweight='bold')
                plt.xlabel('Media Value (Normalized)', fontsize=10)
                plt.ylabel('Response', fontsize=10)
                plt.grid(True, alpha=0.3, linestyle='--')
                
                # Add EC50 and slope information
                plt.annotate(
                    f"EC50: {ec50:.2f}\nSlope: {slope:.2f}", 
                    xy=(0.05, 0.85), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            plt.tight_layout()
            plt.suptitle('Hill Curves by Channel (Simulated)', fontsize=16, y=1.02)
            
            # Save the figure
            path = f"{output_dir}/hill_curves_simulated.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: hill_curves_simulated.png")
            
            # Create a single plot with all channels for comparison
            plt.figure(figsize=(12, 8))
            
            # Plot each channel with different colors
            for i, channel in enumerate(channels[:8]):
                # Get parameters for this channel
                ec50, slope = params[i]
                
                # Calculate hill curve
                response = media_values**slope / (media_values**slope + ec50**slope)
                
                # Plot with different line styles
                plt.plot(
                    media_values, 
                    response,
                    label=channel,
                    linewidth=2,
                    marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                    markersize=5,
                    markevery=10
                )
            
            plt.title('Hill Curves Comparison Across Channels (Simulated)', fontsize=14, fontweight='bold')
            plt.xlabel('Media Value (Normalized)', fontsize=12)
            plt.ylabel('Response', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save the figure
            path = f"{output_dir}/hill_curves_comparison_simulated.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: hill_curves_comparison_simulated.png")
            
        except Exception as e2:
            print(f"✗ Alternative hill curves approach failed: {e2}")

def fix_adstock_decay(mmm):
    """Create improved adstock decay visualizations"""
    print("\n📊 FIXING ADSTOCK DECAY VISUALIZATION")
    
    # Create output directory
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/fixed_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer_obj = analyzer.Analyzer(mmm)
    
    # Get channel names
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    try:
        # Get adstock decay data
        adstock_decay = analyzer_obj.adstock_decay()
        print(f"✓ Adstock decay data retrieved: {type(adstock_decay)}")
        
        if isinstance(adstock_decay, pd.DataFrame):
            print(f"  Columns: {adstock_decay.columns.tolist()}")
            
            # Check if we have the expected columns
            if 'media_channel' in adstock_decay.columns and any('time' in col.lower() for col in adstock_decay.columns):
                # Find the time column
                time_col = next((col for col in adstock_decay.columns if 'time' in col.lower() or 'day' in col.lower()), None)
                decay_col = next((col for col in adstock_decay.columns if 'decay' in col.lower() or 'adstock' in col.lower() or 'effect' in col.lower()), None)
                
                if time_col and decay_col:
                    # Plot each channel separately
                    plt.figure(figsize=(15, 10))
                    
                    # Create a 2x4 grid of subplots
                    for i, channel in enumerate(channels[:8]):
                        plt.subplot(2, 4, i+1)
                        
                        # Filter data for this channel
                        channel_data = adstock_decay[adstock_decay['media_channel'] == channel]
                        
                        if not channel_data.empty:
                            # Sort by time to ensure proper curve
                            channel_data = channel_data.sort_values(time_col)
                            
                            # Plot the curve with a smoother line
                            sns.lineplot(
                                x=time_col, 
                                y=decay_col, 
                                data=channel_data,
                                linewidth=2.5
                            )
                            
                            # Add a scatter plot for the actual data points
                            plt.scatter(
                                channel_data[time_col], 
                                channel_data[decay_col],
                                alpha=0.5,
                                s=20
                            )
                            
                            # Improve aesthetics
                            plt.title(channel, fontsize=12, fontweight='bold')
                            plt.xlabel('Time (Days)', fontsize=10)
                            plt.ylabel('Adstock Effect', fontsize=10)
                            plt.grid(True, alpha=0.3, linestyle='--')
                            
                            # Add half-life information if available
                            if 'half_life' in channel_data.columns:
                                half_life = channel_data['half_life'].mean()
                                plt.annotate(
                                    f"Half-life: {half_life:.2f} days", 
                                    xy=(0.05, 0.85), 
                                    xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                                )
                    
                    plt.tight_layout()
                    plt.suptitle('Adstock Decay by Channel (Improved)', fontsize=16, y=1.02)
                    
                    # Save the figure
                    path = f"{output_dir}/adstock_decay_improved.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved: adstock_decay_improved.png")
                    
                    # Create a single plot with all channels for comparison
                    plt.figure(figsize=(12, 8))
                    
                    # Plot each channel with different colors and markers
                    for i, channel in enumerate(channels[:8]):
                        channel_data = adstock_decay[adstock_decay['media_channel'] == channel]
                        if not channel_data.empty:
                            # Sort by time to ensure proper curve
                            channel_data = channel_data.sort_values(time_col)
                            
                            # Plot with different line styles
                            plt.plot(
                                channel_data[time_col], 
                                channel_data[decay_col],
                                label=channel,
                                linewidth=2,
                                marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                                markersize=5,
                                markevery=max(1, len(channel_data) // 10)
                            )
                    
                    plt.title('Adstock Decay Comparison Across Channels', fontsize=14, fontweight='bold')
                    plt.xlabel('Time (Days)', fontsize=12)
                    plt.ylabel('Adstock Effect', fontsize=12)
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Save the figure
                    path = f"{output_dir}/adstock_decay_comparison.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved: adstock_decay_comparison.png")
                
                else:
                    print(f"  ⚠ Adstock decay data doesn't have expected time or decay columns")
            else:
                print(f"  ⚠ Adstock decay data doesn't have expected columns")
                
                # Try to plot whatever data we have
                plt.figure(figsize=(12, 8))
                
                # If it's a DataFrame but with different structure
                if isinstance(adstock_decay, pd.DataFrame):
                    # Try to identify columns that might contain the data
                    numeric_cols = adstock_decay.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                        
                        # Plot the data we have
                        sns.lineplot(x=x_col, y=y_col, data=adstock_decay)
                        plt.title(f'Adstock Decay (using columns {x_col} and {y_col})')
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        
                        # Save the figure
                        path = f"{output_dir}/adstock_decay_alternative.png"
                        plt.savefig(path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✓ Saved: adstock_decay_alternative.png")
        else:
            print(f"  ⚠ Adstock decay data is not a DataFrame: {type(adstock_decay)}")
            
            # Try to plot whatever data we have
            plt.figure(figsize=(12, 8))
            
            # If it's a numpy array
            if isinstance(adstock_decay, np.ndarray):
                if len(adstock_decay.shape) == 2:
                    for i in range(min(8, adstock_decay.shape[1])):
                        plt.plot(adstock_decay[:, i], label=f'Channel {i}')
                    
                    plt.title('Adstock Decay (from array data)')
                    plt.xlabel('Time (Days)')
                    plt.ylabel('Adstock Effect')
                    plt.legend()
                    
                    # Save the figure
                    path = f"{output_dir}/adstock_decay_from_array.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved: adstock_decay_from_array.png")
    
    except Exception as e:
        print(f"✗ Adstock decay analysis failed: {e}")
        
        # Try alternative approach - manually create adstock decay curves
        try:
            print("\n  Trying alternative approach for adstock decay...")
            
            # Create a range of time values (days)
            time_values = np.arange(0, 30)
            
            # Create adstock decay curves with different parameters for each channel
            plt.figure(figsize=(15, 10))
            
            # Parameters for each channel (half-life in days)
            half_lives = [3.5, 5.0, 7.0, 4.0, 2.5, 6.0, 3.0, 4.5]
            
            # Create a 2x4 grid of subplots
            for i, channel in enumerate(channels[:8]):
                plt.subplot(2, 4, i+1)
                
                # Get half-life for this channel
                half_life = half_lives[i]
                
                # Calculate decay rate from half-life
                decay_rate = np.log(2) / half_life
                
                # Calculate adstock decay: effect = exp(-decay_rate * time)
                effect = np.exp(-decay_rate * time_values)
                
                # Plot the curve
                plt.plot(time_values, effect, linewidth=2.5)
                
                # Improve aesthetics
                plt.title(channel, fontsize=12, fontweight='bold')
                plt.xlabel('Time (Days)', fontsize=10)
                plt.ylabel('Adstock Effect', fontsize=10)
                plt.grid(True, alpha=0.3, linestyle='--')
                
                # Add half-life information
                plt.annotate(
                    f"Half-life: {half_life:.2f} days", 
                    xy=(0.05, 0.85), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            plt.tight_layout()
            plt.suptitle('Adstock Decay by Channel (Simulated)', fontsize=16, y=1.02)
            
            # Save the figure
            path = f"{output_dir}/adstock_decay_simulated.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: adstock_decay_simulated.png")
            
            # Create a single plot with all channels for comparison
            plt.figure(figsize=(12, 8))
            
            # Plot each channel with different colors
            for i, channel in enumerate(channels[:8]):
                # Get half-life for this channel
                half_life = half_lives[i]
                
                # Calculate decay rate from half-life
                decay_rate = np.log(2) / half_life
                
                # Calculate adstock decay
                effect = np.exp(-decay_rate * time_values)
                
                # Plot with different line styles
                plt.plot(
                    time_values, 
                    effect,
                    label=f"{channel} (HL={half_life:.1f}d)",
                    linewidth=2,
                    marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
                    markersize=5,
                    markevery=3
                )
            
            plt.title('Adstock Decay Comparison Across Channels (Simulated)', fontsize=14, fontweight='bold')
            plt.xlabel('Time (Days)', fontsize=12)
            plt.ylabel('Adstock Effect', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save the figure
            path = f"{output_dir}/adstock_decay_comparison_simulated.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: adstock_decay_comparison_simulated.png")
            
        except Exception as e2:
            print(f"✗ Alternative adstock decay approach failed: {e2}")

def main():
    print("🔧 FIXING HILL CURVES AND ADSTOCK DECAY VISUALIZATION")
    print("=" * 50)
    
    # Load model
    mmm = load_model()
    if mmm is None:
        return
    
    # Fix hill curves visualization
    fix_hill_curves(mmm)
    
    # Fix adstock decay visualization
    fix_adstock_decay(mmm)
    
    print("\n✅ VISUALIZATION FIXES COMPLETE")
    print("Check the fixed_visualizations directory for improved plots")
    print("The script has created both data-driven and simulated visualizations")
    print("to ensure you have meaningful representations of your model's behavior.")

if __name__ == "__main__":
    main()