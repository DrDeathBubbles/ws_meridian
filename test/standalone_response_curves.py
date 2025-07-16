#!/usr/bin/env python3
"""
Standalone Response Curves for MMM
=================================
Creates proper hill curves and adstock decay visualizations without dependencies
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_directory():
    """Create output directory for plots"""
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/standalone_curves'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_hill_curves(output_dir):
    """Create proper hill curves visualizations"""
    print("\n📊 CREATING HILL CURVES")
    
    # Define channels
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Define parameters for each channel (EC50, slope)
    # These are typical values that produce proper hill curves
    hill_params = [
        (0.3, 2.0),  # Google
        (0.4, 1.8),  # Facebook
        (0.5, 1.5),  # LinkedIn
        (0.6, 1.2),  # Reddit
        (0.4, 2.2),  # Bing
        (0.5, 1.7),  # TikTok
        (0.6, 1.4),  # Twitter
        (0.7, 1.1)   # Instagram
    ]
    
    # Create a range of media values
    media_values = np.linspace(0, 1, 100)
    
    # Create hill curves with parameters for each channel
    plt.figure(figsize=(15, 10))
    
    # Create a 2x4 grid of subplots
    for i, channel in enumerate(channels[:8]):
        plt.subplot(2, 4, i+1)
        
        # Get parameters for this channel
        ec50, slope = hill_params[i]
        
        # Calculate hill curve: response = media^slope / (media^slope + EC50^slope)
        response = media_values**slope / (media_values**slope + ec50**slope)
        
        # Plot the curve
        plt.plot(media_values, response, linewidth=2.5)
        
        # Add points to make the curve more visible
        plt.scatter(media_values[::10], response[::10], s=30, alpha=0.6)
        
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
        
        # Mark the EC50 point on the curve
        ec50_idx = np.abs(media_values - ec50).argmin()
        plt.scatter([ec50], [response[ec50_idx]], s=100, facecolors='none', 
                   edgecolors='red', linewidth=2)
        plt.axvline(x=ec50, color='red', linestyle='--', alpha=0.3)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Hill Curves by Channel', fontsize=16, y=1.02)
    
    # Save the figure
    path = f"{output_dir}/hill_curves.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curves.png")
    
    # Create a single plot with all channels for comparison
    plt.figure(figsize=(12, 8))
    
    # Plot each channel with different colors
    for i, channel in enumerate(channels[:8]):
        # Get parameters for this channel
        ec50, slope = hill_params[i]
        
        # Calculate hill curve
        response = media_values**slope / (media_values**slope + ec50**slope)
        
        # Plot with different line styles
        plt.plot(
            media_values, 
            response,
            label=f"{channel} (EC50={ec50:.2f}, S={slope:.1f})",
            linewidth=2,
            marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
            markersize=5,
            markevery=10
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
    
    return True

def create_adstock_decay(output_dir):
    """Create proper adstock decay visualizations"""
    print("\n📊 CREATING ADSTOCK DECAY CURVES")
    
    # Define channels
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Define half-life parameters for each channel (in days)
    # These are typical values that produce proper adstock decay curves
    half_lives = [3.5, 5.0, 7.0, 4.0, 2.5, 6.0, 3.0, 4.5]
    
    # Create a range of time values (days)
    time_values = np.arange(0, 30)
    
    # Create adstock decay curves with parameters for each channel
    plt.figure(figsize=(15, 10))
    
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
        
        # Add points to make the curve more visible
        plt.scatter(time_values[::3], effect[::3], s=30, alpha=0.6)
        
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
        
        # Mark the half-life point on the curve
        plt.scatter([half_life], [0.5], s=100, facecolors='none', 
                   edgecolors='red', linewidth=2)
        plt.axvline(x=half_life, color='red', linestyle='--', alpha=0.3)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Adstock Decay by Channel', fontsize=16, y=1.02)
    
    # Save the figure
    path = f"{output_dir}/adstock_decay.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay.png")
    
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
    
    return True

def create_combined_effects(output_dir):
    """Create visualizations showing combined effects"""
    print("\n📊 CREATING COMBINED EFFECTS VISUALIZATION")
    
    # Create figure showing the combined effect
    plt.figure(figsize=(12, 8))
    
    # Create a simple grid layout
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
    
    # 1. Hill Curve (top left)
    ax1 = plt.subplot(gs[0, 0])
    
    # Parameters
    media_values = np.linspace(0, 1, 100)
    ec50 = 0.5
    slope = 2.0
    
    # Calculate hill curve
    response = media_values**slope / (media_values**slope + ec50**slope)
    
    # Plot the curve
    ax1.plot(media_values, response, linewidth=2.5, color='blue')
    ax1.set_title('1. Hill Curve (Diminishing Returns)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Media Value', fontsize=10)
    ax1.set_ylabel('Immediate Response', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Adstock Decay (top right)
    ax2 = plt.subplot(gs[0, 1])
    
    # Parameters
    time_values = np.arange(0, 10)
    half_life = 3
    decay_rate = np.log(2) / half_life
    
    # Calculate adstock decay
    effect = np.exp(-decay_rate * time_values)
    
    # Plot the curve
    ax2.plot(time_values, effect, linewidth=2.5, color='green')
    ax2.set_title('2. Adstock Decay (Carryover)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (Days)', fontsize=10)
    ax2.set_ylabel('Decay Factor', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Combined Effect (bottom)
    ax3 = plt.subplot(gs[1, :])
    
    # Create time and media value grids
    T, M = np.meshgrid(time_values, media_values[:20])
    
    # Calculate combined effect
    hill_response = M**slope / (M**slope + ec50**slope)
    adstock_effect = np.exp(-decay_rate * T)
    combined = hill_response[:, np.newaxis] * adstock_effect
    
    # Plot as a heatmap
    im = ax3.imshow(combined, aspect='auto', origin='lower', 
                   extent=[time_values.min(), time_values.max(), 
                          media_values[:20].min(), media_values[:20].max()],
                   cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Response Magnitude')
    
    ax3.set_title('3. Combined Effect (Hill Curve × Adstock Decay)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (Days)', fontsize=10)
    ax3.set_ylabel('Media Value', fontsize=10)
    
    plt.tight_layout()
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'In MMM, the hill curve determines the immediate response to media spend,\n'
                'while adstock decay determines how that response persists over time.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/combined_effects.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: combined_effects.png")
    
    return True

def main():
    print("🔧 CREATING STANDALONE RESPONSE CURVES")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Create hill curves
    create_hill_curves(output_dir)
    
    # Create adstock decay curves
    create_adstock_decay(output_dir)
    
    # Create combined effects visualization
    create_combined_effects(output_dir)
    
    print("\n✅ VISUALIZATIONS COMPLETE")
    print("Check the standalone_curves directory for the new visualizations")
    print("These curves provide proper examples of what hill curves and")
    print("adstock decay functions should look like in your MMM model.")
    print("\nTo fix your model's curves:")
    print("1. Compare these proper curves with your model's current output")
    print("2. Check your model's prior distributions for hill and adstock parameters")
    print("3. Ensure your model is properly converging during MCMC sampling")
    print("4. Consider using stronger priors to guide the model toward proper curves")

if __name__ == "__main__":
    main()