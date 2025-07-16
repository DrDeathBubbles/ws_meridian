#!/usr/bin/env python3
"""
Data Diagnostic Script
=====================
Examines the CSV data and Meridian data loading process
"""

import pandas as pd
import numpy as np
from meridian.data import load

def main():
    print("🔍 DATA DIAGNOSTIC")
    print("=" * 50)
    
    # 1. Read raw CSV
    csv_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data.csv'
    df = pd.read_csv(csv_path)
    
    print("1. RAW CSV DATA:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check spend columns
    spend_cols = ['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 
                  'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend']
    
    print("\n2. SPEND COLUMN ANALYSIS:")
    for col in spend_cols:
        if col in df.columns:
            total = df[col].sum()
            non_zero = (df[col] != 0).sum()
            print(f"   {col}: Total={total:.2f}, Non-zero rows={non_zero}")
        else:
            print(f"   {col}: MISSING")
    
    # Check impression columns  
    impression_cols = ['google_impressions', 'facebook_impressions', 'linkedin_impressions', 
                      'reddit_impressions', 'bing_impressions', 'tiktok_impressions', 
                      'twitter_impressions', 'instagram_impressions']
    
    print("\n3. IMPRESSION COLUMN ANALYSIS:")
    for col in impression_cols:
        if col in df.columns:
            total = df[col].sum()
            non_zero = (df[col] != 0).sum()
            print(f"   {col}: Total={total:.0f}, Non-zero rows={non_zero}")
        else:
            print(f"   {col}: MISSING")
    
    # 4. Load with Meridian
    print("\n4. MERIDIAN DATA LOADING:")
    try:
        coord_to_columns = load.CoordToColumns(
            time='date_id',
            geo='national_geo',
            kpi='clicks',
            media_spend=spend_cols,
            media=impression_cols,
        )
        
        loader = load.CsvDataLoader(
            csv_path=csv_path,
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
        
        data = loader.load()
        print("   ✓ Meridian loading successful")
        print(f"   Media spend shape: {data.media_spend.shape}")
        print(f"   Media spend total: {np.sum(data.media_spend):.2f}")
        print(f"   Media shape: {data.media.shape}")
        print(f"   Media total: {np.sum(data.media):.0f}")
        
    except Exception as e:
        print(f"   ✗ Meridian loading failed: {e}")
    
    # 5. Sample data inspection
    print("\n5. SAMPLE DATA (first 5 rows):")
    sample_cols = ['date_id', 'national_geo', 'clicks'] + spend_cols[:3]
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head())
    
    print("\n6. RECOMMENDATIONS:")
    if all(df[col].sum() == 0 for col in spend_cols if col in df.columns):
        print("   ⚠ All spend columns are zero!")
        print("   - Check data processing pipeline")
        print("   - Verify original data source")
        print("   - Consider using impression data instead")
    else:
        print("   ✓ Spend data looks normal")

if __name__ == "__main__":
    main()