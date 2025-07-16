#!/usr/bin/env python3
"""
Check for negative values in ticket sales data
"""

import pandas as pd
import numpy as np

def check_data():
    print("🔍 CHECKING FOR NEGATIVE VALUES")
    print("=" * 50)
    
    # Load merged data
    data_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/merged_marketing_tickets.csv'
    df = pd.read_csv(data_path)
    
    print(f"Data loaded: {df.shape}")
    
    # Check ticket sales for negative values
    if 'ticket_sales' in df.columns:
        neg_count = (df['ticket_sales'] < 0).sum()
        print(f"Negative ticket sales: {neg_count} rows")
        
        if neg_count > 0:
            print("\nSample of negative values:")
            neg_samples = df[df['ticket_sales'] < 0][['date_id', 'ticket_sales']].head(10)
            print(neg_samples)
            
            # Fix negative values
            print("\nFixing negative values...")
            df['ticket_sales_fixed'] = df['ticket_sales'].clip(lower=0)
            
            # Save fixed data
            fixed_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/merged_marketing_tickets_fixed.csv'
            df.to_csv(fixed_path, index=False)
            print(f"Fixed data saved to: {fixed_path}")
            
            # Verify fix
            print(f"Original min value: {df['ticket_sales'].min()}")
            print(f"Fixed min value: {df['ticket_sales_fixed'].min()}")
            
            # Replace original column
            df['ticket_sales'] = df['ticket_sales_fixed']
            df = df.drop(columns=['ticket_sales_fixed'])
            df.to_csv(fixed_path, index=False)
            print(f"Final data saved with fixed values")
        else:
            print("✓ No negative values found in ticket_sales")
    else:
        print("❌ No 'ticket_sales' column found")
        
    # Check other columns that should be non-negative
    for col in df.columns:
        if 'spend' in col.lower() or 'impression' in col.lower():
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"Column {col}: {neg_count} negative values")
    
    print("\nDONE")

if __name__ == "__main__":
    check_data()