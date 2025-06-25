#!/usr/bin/env python3
"""
Working Meridian Example - Data Loading Only
"""

import numpy as np
import pandas as pd
from meridian.data import load

def main():
    print("=" * 50)
    print("MERIDIAN DATA LOADING TEST")
    print("=" * 50)
    
    try:
        # Check data first
        df = pd.read_csv('/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data.csv')
        print(f"✓ Data loaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Setup column mappings
        coord_to_columns = load.CoordToColumns(
            time='date_id',
            geo='national_geo', 
            kpi='clicks',
            media_spend=['google_spend', 'facebook_spend', 'linkedin_spend'],
            media=['google_impressions', 'facebook_impressions', 'linkedin_impressions'],
        )
        
        media_to_channel = {
            'google_impressions': 'Google',
            'facebook_impressions': 'Facebook',
            'linkedin_impressions': 'LinkedIn'
        }
        
        media_spend_to_channel = {
            'google_spend': 'Google', 
            'facebook_spend': 'Facebook',
            'linkedin_spend': 'LinkedIn'
        }
        
        # Load with Meridian
        loader = load.CsvDataLoader(
            csv_path='/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data.csv',
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=media_to_channel,
            media_spend_to_channel=media_spend_to_channel,
        )
        
        data = loader.load()
        print("✓ Meridian data loading successful!")
        print(f"KPI shape: {data.kpi.shape}")
        print(f"Media shape: {data.media.shape}")
        print(f"Media spend shape: {data.media_spend.shape}")
        
        # Data summary
        print(f"\nData Summary:")
        print(f"Total clicks: {np.sum(data.kpi):.0f}")
        print(f"Total spend: {np.sum(data.media_spend):.2f}")
        print(f"Date range: {data.time.min()} to {data.time.max()}")
        
        print("\n✓ DATA LOADING WORKS!")
        print("Note: Model fitting has known tensor type issues in current Meridian version")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()