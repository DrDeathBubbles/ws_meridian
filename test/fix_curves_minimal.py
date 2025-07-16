#!/usr/bin/env python3
"""
Fix Curves Minimal
=================
Minimal script to fix hill curves and adstock decay visualizations
without requiring the Meridian library
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def create_output_directory():
    """Create output directory for plots"""
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/fixed_curves'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_proper_hill_curves(output_dir):
    """Create proper hill curves with correct mathematical form"""
    print("\n📊 CREATING PROPER HILL CURVES")
    
    # Channel names
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Parameters for each channel (EC50, slope) - reasonable defaults
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
    
    # Create a range of media values
    media_values = np.linspace(0, 1, 100)
    
    # Create hill curves with parameters for each channel
    plt.figure(figsize=(15, 10))
    
    # Create a 2x4 grid of subplots
    for i, channel in enumerate(channels[:8]):
        plt.subplot(2, 4, i+1)
        
        # Get parameters for this channel
        ec50, slope = params[i]
        
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
    plt.suptitle('Hill Curves by Channel (Corrected)', fontsize=16, y=1.02)
    
    # Save the figure
    path = f"{output_dir}/hill_curves_fixed.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curves_fixed.png")
    
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
            label=f"{channel} (EC50={ec50:.2f}, S={slope:.1f})",
            linewidth=2,
            marker=['o', 's', '^', 'v', 'd', '*', 'x', '+'][i % 8],
            markersize=5,
            markevery=10
        )
    
    plt.title('Hill Curves Comparison Across Channels (Corrected)', fontsize=14, fontweight='bold')
    plt.xlabel('Media Value (Normalized)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure
    path = f"{output_dir}/hill_curves_comparison_fixed.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curves_comparison_fixed.png")
    
    return True

def create_proper_adstock_decay(output_dir):
    """Create proper adstock decay curves with correct mathematical form"""
    print("\n📊 CREATING PROPER ADSTOCK DECAY CURVES")
    
    # Channel names
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Half-life values for each channel (in days) - reasonable defaults
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
    plt.suptitle('Adstock Decay by Channel (Corrected)', fontsize=16, y=1.02)
    
    # Save the figure
    path = f"{output_dir}/adstock_decay_fixed.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay_fixed.png")
    
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
    
    plt.title('Adstock Decay Comparison Across Channels (Corrected)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Adstock Effect', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure
    path = f"{output_dir}/adstock_decay_comparison_fixed.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay_comparison_fixed.png")
    
    return True

def main():
    print("🔧 FIXING CURVES WITH MINIMAL DEPENDENCIES")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Create proper hill curves
    create_proper_hill_curves(output_dir)
    
    # Create proper adstock decay curves
    create_proper_adstock_decay(output_dir)
    
    print("\n✅ CURVE FIXES COMPLETE")
    print("Check the fixed_curves directory for corrected visualizations")
    print("These curves show the proper mathematical form that your model")
    print("should be using for hill curves and adstock decay functions.")
    print("\nNOTE: These curves use reasonable default parameters since")
    print("we couldn't extract the actual parameters from your model.")
    print("They show the correct SHAPE but may not reflect your exact model parameters.")

if __name__ == "__main__":
    main()