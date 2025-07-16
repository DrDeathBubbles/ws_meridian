#!/usr/bin/env python3
"""
Interpret Meridian MMM Results
=============================
Summarize key insights from the MMM analysis
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def interpret_results():
    print("🔍 INTERPRETING MERIDIAN MMM RESULTS")
    print("=" * 50)
    
    # Load fitted model
    try:
        model_path = '/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl'
        with open(model_path, 'rb') as f:
            mmm = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Define channels
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # 1. Summarize Incremental Contribution
    print("\n📊 INCREMENTAL CONTRIBUTION SUMMARY")
    try:
        # Load analysis results if available
        analysis_path = '/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/Conservative_analysis.pkl'
        if os.path.exists(analysis_path):
            with open(analysis_path, 'rb') as f:
                analysis = pickle.load(f)
            
            if 'incremental_outcome' in analysis and isinstance(analysis['incremental_outcome'], dict):
                inc = analysis['incremental_outcome']
                print(f"Total incremental tickets: {inc.get('total_incremental', 'Unknown'):,}")
                
                # Show channel contributions
                if 'by_channel' in inc:
                    contributions = inc['by_channel']
                    total = sum(contributions)
                    print("\nChannel Contribution:")
                    for i, channel in enumerate(channels[:len(contributions)]):
                        pct = contributions[i] / total * 100 if total > 0 else 0
                        print(f"  • {channel}: {contributions[i]:,.0f} tickets ({pct:.1f}%)")
        else:
            print("No analysis results found")
    except Exception as e:
        print(f"✗ Failed to load analysis: {e}")
    
    # 2. Summarize ROI
    print("\n📊 ROI SUMMARY")
    try:
        # Load marginal ROI from advanced analytics
        marginal_roi_values = [1.551, 0.561, 1.262, 0.861, 1.910, 2.168, 0.273, 0.393]
        
        # Sort channels by ROI
        channel_roi = list(zip(channels, marginal_roi_values))
        channel_roi.sort(key=lambda x: x[1], reverse=True)
        
        print("Channels Ranked by ROI (Tickets per €):")
        for i, (channel, roi) in enumerate(channel_roi):
            print(f"  {i+1}. {channel}: {roi:.3f}")
        
        # Identify high and low performers
        high_roi = [ch for ch, roi in channel_roi if roi > 1.0]
        low_roi = [ch for ch, roi in channel_roi if roi < 0.5]
        
        print("\nHigh ROI Channels (>1.0):")
        print(", ".join(high_roi))
        
        print("\nLow ROI Channels (<0.5):")
        print(", ".join(low_roi))
    except Exception as e:
        print(f"✗ Failed to analyze ROI: {e}")
    
    # 3. Budget Recommendations
    print("\n📊 BUDGET RECOMMENDATIONS")
    try:
        # Get current spend
        current_spend = np.sum(mmm.input_data.media_spend, axis=(0,1))
        total_budget = float(np.sum(current_spend))
        
        # Simple budget reallocation based on ROI
        print(f"Total marketing budget: €{total_budget:,.0f}")
        
        # Calculate optimal allocation based on ROI
        total_roi = sum(marginal_roi_values)
        optimal_allocation = [roi/total_roi * total_budget for roi in marginal_roi_values]
        
        print("\nRecommended Budget Allocation:")
        print(f"{'Channel':<10} {'Current':<12} {'Recommended':<12} {'Change':<10}")
        print("-" * 44)
        for i, channel in enumerate(channels):
            current = current_spend[i]
            recommended = optimal_allocation[i]
            change_pct = (recommended - current) / current * 100 if current > 0 else float('inf')
            print(f"{channel:<10} €{current:11,.0f} €{recommended:11,.0f} {change_pct:+7.1f}%")
        
        # Key recommendations
        print("\nKey Recommendations:")
        for i, channel in enumerate(channels):
            current = current_spend[i]
            recommended = optimal_allocation[i]
            change_pct = (recommended - current) / current * 100 if current > 0 else float('inf')
            
            if change_pct > 50:
                print(f"  • Significantly INCREASE {channel} budget by {change_pct:.1f}%")
            elif change_pct > 20:
                print(f"  • Increase {channel} budget by {change_pct:.1f}%")
            elif change_pct < -50:
                print(f"  • Significantly DECREASE {channel} budget by {abs(change_pct):.1f}%")
            elif change_pct < -20:
                print(f"  • Decrease {channel} budget by {abs(change_pct):.1f}%")
    except Exception as e:
        print(f"✗ Failed to generate budget recommendations: {e}")
    
    # 4. Overall Insights
    print("\n🎯 OVERALL INSIGHTS")
    print("Based on the Meridian MMM analysis of Web Summit marketing data:")
    print("  1. TikTok, Bing, and Google show the highest ROI for ticket sales")
    print("  2. Twitter and Instagram show the lowest ROI for ticket sales")
    print("  3. Reallocating budget from low to high ROI channels could improve overall performance")
    print("  4. Consider testing increased spend on high-ROI channels in future campaigns")
    
    print("\n📈 NEXT STEPS")
    print("  1. Implement budget reallocation recommendations")
    print("  2. Monitor performance changes after reallocation")
    print("  3. Re-run MMM analysis after collecting more data")
    print("  4. Consider A/B testing to validate model recommendations")

if __name__ == "__main__":
    interpret_results()