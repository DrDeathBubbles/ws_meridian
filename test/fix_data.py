#!/usr/bin/env python3
"""
Fix Data Processing
==================
Properly distribute spend and impressions to channel columns
"""

import pandas as pd
import numpy as np

def fix_data():
    # Load the original raw data
    raw_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/advertising_raw.csv'
    df = pd.read_csv(raw_path)
    
    print("Original data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Check if we have platform information
    if 'marketing_platform_name' in df.columns:
        print("\nPlatform distribution:")
        print(df['marketing_platform_name'].value_counts())
        
        # Process the data properly
        platforms = ['reddit', 'linkedin', 'facebook', 'google', 'bing', 'tiktok', 'twitter', 'instagram']
        
        # Initialize spend and impression columns
        for platform in platforms:
            df[f'{platform}_spend'] = 0.0
            df[f'{platform}_impressions'] = 0.0
        
        # Platform name mapping
        platform_mapping = {
            'FacebookAds': 'facebook',
            'GoogleAds': 'google', 
            'LinkedInAds': 'linkedin',
            'BingAds': 'bing',
            'TikTok': 'tiktok',
            'TwitterAds': 'twitter',
            'Instagram': 'instagram',
            'Reddit': 'reddit'
        }
        
        # Distribute spend and impressions by platform
        for idx, row in df.iterrows():
            platform_name = str(row['marketing_platform_name'])
            if platform_name in platform_mapping:
                platform = platform_mapping[platform_name]
                df.at[idx, f'{platform}_spend'] = row.get('campaign_spend_eur', 0)
                df.at[idx, f'{platform}_impressions'] = row.get('impressions', 0)
        
        # Group by date and sum
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df_grouped = df.groupby('date_id')[numeric_columns].sum().reset_index()
        
        # Add back non-numeric columns
        df_grouped['national_geo'] = 'TOTAL'
        
        # Save fixed data
        output_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv'
        df_grouped.to_csv(output_path, index=False)
        
        print(f"\nFixed data saved to: {output_path}")
        
        # Verify the fix
        spend_cols = [f'{p}_spend' for p in platforms]
        impression_cols = [f'{p}_impressions' for p in platforms]
        
        print("\nSpend verification:")
        for col in spend_cols:
            if col in df_grouped.columns:
                total = df_grouped[col].sum()
                print(f"  {col}: {total:.2f}")
        
        print("\nImpression verification:")
        for col in impression_cols:
            if col in df_grouped.columns:
                total = df_grouped[col].sum()
                print(f"  {col}: {total:.0f}")
                
        return output_path
    
    else:
        print("No marketing_platform_name column found!")
        return None

if __name__ == "__main__":
    fix_data()