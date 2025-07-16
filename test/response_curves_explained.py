#!/usr/bin/env python3
"""
Response Curves Explained
========================
Educational script explaining hill curves and adstock decay functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_directory():
    """Create output directory for plots"""
    output_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/response_curves_explained'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def explain_hill_curves(output_dir):
    """Explain hill curves with educational visualizations"""
    print("\n📊 HILL CURVES EXPLAINED")
    print("=" * 50)
    print("Hill curves model the diminishing returns of media spend.")
    print("As you increase spend, the incremental impact decreases.")
    
    # Create a range of media values
    media_values = np.linspace(0, 1, 100)
    
    # Create figure for different EC50 values
    plt.figure(figsize=(10, 6))
    
    # Plot hill curves with different EC50 values
    ec50_values = [0.2, 0.4, 0.6, 0.8]
    slope = 2.0  # Fixed slope
    
    for ec50 in ec50_values:
        # Calculate hill curve: response = media^slope / (media^slope + EC50^slope)
        response = media_values**slope / (media_values**slope + ec50**slope)
        
        # Plot the curve
        plt.plot(media_values, response, linewidth=2.5, label=f'EC50 = {ec50}')
    
    # Improve aesthetics
    plt.title('Hill Curves with Different EC50 Values', fontsize=14, fontweight='bold')
    plt.xlabel('Media Value (Normalized)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Parameter')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'EC50 represents the media value that produces 50% of the maximum response.\n'
                'Lower EC50 values mean the channel is more efficient at lower spend levels.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/hill_curves_ec50.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curves_ec50.png")
    
    # Create figure for different slope values
    plt.figure(figsize=(10, 6))
    
    # Plot hill curves with different slope values
    slope_values = [0.5, 1.0, 2.0, 4.0]
    ec50 = 0.5  # Fixed EC50
    
    for slope in slope_values:
        # Calculate hill curve: response = media^slope / (media^slope + EC50^slope)
        response = media_values**slope / (media_values**slope + ec50**slope)
        
        # Plot the curve
        plt.plot(media_values, response, linewidth=2.5, label=f'Slope = {slope}')
    
    # Improve aesthetics
    plt.title('Hill Curves with Different Slope Values', fontsize=14, fontweight='bold')
    plt.xlabel('Media Value (Normalized)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Parameter')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'Slope controls how quickly the response changes around the EC50 point.\n'
                'Higher slope values create a steeper S-curve with more pronounced diminishing returns.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/hill_curves_slope.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curves_slope.png")
    
    # Create figure showing the key components of a hill curve
    plt.figure(figsize=(10, 6))
    
    # Parameters
    ec50 = 0.5
    slope = 2.0
    
    # Calculate hill curve
    response = media_values**slope / (media_values**slope + ec50**slope)
    
    # Plot the curve
    plt.plot(media_values, response, linewidth=3, color='blue')
    
    # Mark the EC50 point
    plt.scatter([ec50], [0.5], s=100, color='red', zorder=5)
    plt.axvline(x=ec50, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations
    plt.annotate('EC50 Point (50% Response)', 
                xy=(ec50, 0.5), xytext=(ec50+0.1, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    plt.annotate('Diminishing Returns Region', 
                xy=(0.7, 0.8), xytext=(0.7, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    plt.annotate('Linear Growth Region', 
                xy=(0.2, 0.2), xytext=(0.05, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    # Improve aesthetics
    plt.title('Anatomy of a Hill Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Media Value (Normalized)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'The Hill curve shows how media response changes with increasing media value.\n'
                'It captures both the initial linear growth and the eventual saturation (diminishing returns).',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/hill_curve_anatomy.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: hill_curve_anatomy.png")
    
    # Print explanation
    print("\nHILL CURVES EXPLANATION:")
    print("------------------------")
    print("1. Hill curves model the relationship between media spend and response")
    print("2. They capture the diminishing returns effect of advertising")
    print("3. Key parameters:")
    print("   - EC50: Media value that produces 50% of maximum response")
    print("   - Slope: Controls how quickly the response changes around EC50")
    print("4. A proper hill curve should be S-shaped, not a saw-tooth pattern")
    print("5. Lower EC50 values indicate more efficient channels at lower spend")
    print("6. Higher slope values indicate more pronounced diminishing returns")

def explain_adstock_decay(output_dir):
    """Explain adstock decay with educational visualizations"""
    print("\n📊 ADSTOCK DECAY EXPLAINED")
    print("=" * 50)
    print("Adstock decay models how media effects persist over time.")
    print("It captures the carryover effect of advertising.")
    
    # Create a range of time values (days)
    time_values = np.arange(0, 30)
    
    # Create figure for different half-life values
    plt.figure(figsize=(10, 6))
    
    # Plot adstock decay curves with different half-life values
    half_life_values = [2, 4, 7, 14]
    
    for half_life in half_life_values:
        # Calculate decay rate from half-life
        decay_rate = np.log(2) / half_life
        
        # Calculate adstock decay: effect = exp(-decay_rate * time)
        effect = np.exp(-decay_rate * time_values)
        
        # Plot the curve
        plt.plot(time_values, effect, linewidth=2.5, label=f'Half-life = {half_life} days')
    
    # Improve aesthetics
    plt.title('Adstock Decay with Different Half-life Values', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Adstock Effect', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Parameter')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'Half-life represents the time it takes for the media effect to decay to 50% of its initial value.\n'
                'Longer half-life values mean the channel has more persistent effects over time.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/adstock_decay_half_life.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay_half_life.png")
    
    # Create figure showing the key components of adstock decay
    plt.figure(figsize=(10, 6))
    
    # Parameters
    half_life = 7
    
    # Calculate decay rate from half-life
    decay_rate = np.log(2) / half_life
    
    # Calculate adstock decay
    effect = np.exp(-decay_rate * time_values)
    
    # Plot the curve
    plt.plot(time_values, effect, linewidth=3, color='blue')
    
    # Mark the half-life point
    plt.scatter([half_life], [0.5], s=100, color='red', zorder=5)
    plt.axvline(x=half_life, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations
    plt.annotate('Half-life Point (50% Effect)', 
                xy=(half_life, 0.5), xytext=(half_life+2, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    plt.annotate('Immediate Effect (100%)', 
                xy=(0, 1.0), xytext=(2, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    plt.annotate('Long-term Decay', 
                xy=(20, 0.1), xytext=(15, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    # Improve aesthetics
    plt.title('Anatomy of Adstock Decay', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Adstock Effect', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'Adstock decay shows how media effects persist over time after the initial exposure.\n'
                'It should be a smooth exponential decay curve, not a saw-tooth pattern.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/adstock_decay_anatomy.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay_anatomy.png")
    
    # Create figure comparing different decay functions
    plt.figure(figsize=(10, 6))
    
    # Parameters
    half_life = 7
    decay_rate = np.log(2) / half_life
    
    # Calculate different decay functions
    exponential_decay = np.exp(-decay_rate * time_values)
    geometric_decay = (1 - decay_rate/2) ** time_values
    weibull_decay = np.exp(-(time_values / half_life) ** 1.5)
    
    # Plot the curves
    plt.plot(time_values, exponential_decay, linewidth=2.5, label='Exponential Decay')
    plt.plot(time_values, geometric_decay, linewidth=2.5, label='Geometric Decay')
    plt.plot(time_values, weibull_decay, linewidth=2.5, label='Weibull Decay')
    
    # Improve aesthetics
    plt.title('Different Types of Adstock Decay Functions', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Adstock Effect', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Decay Type')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                'Different mathematical functions can model adstock decay.\n'
                'Exponential decay is most common, but other functions may better fit specific channels.',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', alpha=0.8))
    
    # Save the figure
    path = f"{output_dir}/adstock_decay_types.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: adstock_decay_types.png")
    
    # Print explanation
    print("\nADSTOCK DECAY EXPLANATION:")
    print("--------------------------")
    print("1. Adstock decay models how media effects persist over time")
    print("2. It captures the carryover effect of advertising")
    print("3. Key parameter:")
    print("   - Half-life: Time it takes for the effect to decay to 50%")
    print("4. A proper adstock decay should be a smooth exponential curve, not a saw-tooth pattern")
    print("5. Longer half-life values indicate more persistent media effects")
    print("6. Different channels typically have different decay rates")
    print("7. Common decay functions include exponential, geometric, and Weibull")

def explain_combined_effects(output_dir):
    """Explain how hill curves and adstock decay work together"""
    print("\n📊 COMBINED EFFECTS EXPLAINED")
    print("=" * 50)
    print("Hill curves and adstock decay work together in MMM models.")
    
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
    
    # Print explanation
    print("\nCOMBINED EFFECTS EXPLANATION:")
    print("----------------------------")
    print("1. In MMM, hill curves and adstock decay work together:")
    print("   - Hill curve: Determines the immediate response to media spend")
    print("   - Adstock decay: Determines how that response persists over time")
    print("2. The total media effect is the product of these two components")
    print("3. This allows the model to capture both:")
    print("   - Diminishing returns (efficiency decreases as spend increases)")
    print("   - Carryover effects (impact continues after the initial exposure)")
    print("4. Different channels can have different parameters for both components")
    print("5. Optimizing media spend requires understanding both components")

def main():
    print("📚 RESPONSE CURVES EXPLAINED")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Explain hill curves
    explain_hill_curves(output_dir)
    
    # Explain adstock decay
    explain_adstock_decay(output_dir)
    
    # Explain combined effects
    explain_combined_effects(output_dir)
    
    print("\n✅ EDUCATIONAL MATERIALS CREATED")
    print("Check the response_curves_explained directory for visualizations")
    print("These materials explain what proper hill curves and adstock decay")
    print("functions should look like and how they work in MMM models.")

if __name__ == "__main__":
    main()