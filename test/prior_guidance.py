#!/usr/bin/env python3
"""
Prior Selection Guidance
=======================
Calculate data-driven priors for your Web Summit data
"""

import pandas as pd
import numpy as np

def calculate_priors():
    # Load your fixed data
    df = pd.read_csv('/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv')
    
    print("📊 DATA-DRIVEN PRIOR RECOMMENDATIONS")
    print("=" * 50)
    
    # 1. Calculate empirical ROI by channel
    channels = ['google', 'facebook', 'linkedin', 'reddit', 'bing', 'tiktok', 'twitter', 'instagram']
    
    total_clicks = df['clicks'].sum()
    total_spend = sum(df[f'{ch}_spend'].sum() for ch in channels)
    
    print(f"Total clicks: {total_clicks:,.0f}")
    print(f"Total spend: €{total_spend:,.0f}")
    print(f"Overall efficiency: {total_clicks/total_spend:.2f} clicks/€")
    
    print("\n1. CHANNEL-SPECIFIC ROI ANALYSIS:")
    roi_values = []
    for channel in channels:
        spend = df[f'{channel}_spend'].sum()
        impressions = df[f'{channel}_impressions'].sum()
        
        if spend > 0:
            # ROI as clicks per euro spent
            channel_clicks = df['clicks'].sum() * (impressions / df['impressions'].sum()) if df['impressions'].sum() > 0 else 0
            roi = channel_clicks / spend if spend > 0 else 0
            roi_values.append(roi)
            print(f"  {channel.capitalize()}: €{spend:,.0f} → {roi:.3f} clicks/€")
        else:
            print(f"  {channel.capitalize()}: No spend data")
    
    # 2. ROI Prior Recommendations
    print("\n2. ROI PRIOR RECOMMENDATIONS:")
    if roi_values:
        median_roi = np.median(roi_values)
        mean_roi = np.mean(roi_values)
        std_roi = np.std(roi_values)
        
        print(f"  Empirical ROI - Mean: {mean_roi:.3f}, Median: {median_roi:.3f}, Std: {std_roi:.3f}")
        
        # LogNormal parameters
        conservative_loc = np.log(median_roi * 0.5)  # 50% of observed
        moderate_loc = np.log(median_roi)            # Observed median
        optimistic_loc = np.log(median_roi * 1.5)    # 150% of observed
        
        print(f"\n  RECOMMENDED PRIORS:")
        print(f"  Conservative: LogNormal(loc={conservative_loc:.3f}, scale=0.3)")
        print(f"  Moderate:     LogNormal(loc={moderate_loc:.3f}, scale=0.5)")
        print(f"  Optimistic:   LogNormal(loc={optimistic_loc:.3f}, scale=0.7)")
    
    # 3. Spend Distribution Analysis
    print("\n3. SPEND DISTRIBUTION:")
    spend_by_channel = {}
    for channel in channels:
        spend = df[f'{channel}_spend'].sum()
        spend_by_channel[channel] = spend
    
    total_spend = sum(spend_by_channel.values())
    for channel, spend in sorted(spend_by_channel.items(), key=lambda x: x[1], reverse=True):
        if spend > 0:
            pct = spend / total_spend * 100
            print(f"  {channel.capitalize()}: €{spend:,.0f} ({pct:.1f}%)")
    
    # 4. Time Series Patterns
    print("\n4. SEASONALITY INSIGHTS:")
    df['date_id'] = pd.to_datetime(df['date_id'])
    df['month'] = df['date_id'].dt.month
    monthly_spend = df.groupby('month')[f'clicks'].sum()
    peak_month = monthly_spend.idxmax()
    low_month = monthly_spend.idxmin()
    
    print(f"  Peak activity: Month {peak_month}")
    print(f"  Low activity: Month {low_month}")
    print(f"  Seasonality factor: {monthly_spend.max()/monthly_spend.min():.1f}x")
    
    # 5. Final Recommendations
    print("\n5. 🎯 FINAL RECOMMENDATIONS:")
    print("  Based on your Web Summit data:")
    print(f"  • Use MODERATE priors: LogNormal(loc={moderate_loc:.3f}, scale=0.5)")
    print("  • LinkedIn is your biggest spender - expect higher contribution")
    print("  • Facebook has highest volume - may show strong effects")
    print("  • Consider seasonal priors if modeling across months")
    print("  • Start with tighter priors (scale=0.3) for better convergence")
    
    return {
        'conservative': {'loc': conservative_loc, 'scale': 0.3},
        'moderate': {'loc': moderate_loc, 'scale': 0.5},
        'optimistic': {'loc': optimistic_loc, 'scale': 0.7}
    }

if __name__ == "__main__":
    priors = calculate_priors()