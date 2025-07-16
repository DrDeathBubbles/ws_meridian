#!/usr/bin/env python3
"""
Merge Ticket Sales Data
=======================
Combine marketing data with ticket sales for Meridian analysis
"""

import pandas as pd
import numpy as np

def merge_data():
    print("🔗 MERGING MARKETING AND TICKET SALES DATA")
    print("=" * 50)
    
    # Load marketing data
    marketing_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv'
    marketing_df = pd.read_csv(marketing_path)
    print(f"✓ Marketing data loaded: {marketing_df.shape}")
    print(f"  Date range: {marketing_df['date_id'].min()} to {marketing_df['date_id'].max()}")
    
    # Load ticket sales data
    ticket_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/ticket_sales_raw.csv'
    ticket_df = pd.read_csv(ticket_path)
    print(f"✓ Ticket data loaded: {ticket_df.shape}")
    print(f"  Columns: {list(ticket_df.columns)}")
    
    # Check ticket data structure
    if 'date' in ticket_df.columns:
        date_col = 'date'
    elif 'date_id' in ticket_df.columns:
        date_col = 'date_id'
    else:
        # Find date-like column
        date_cols = [col for col in ticket_df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
        else:
            print("❌ No date column found in ticket data")
            return None
    
    print(f"  Using date column: {date_col}")
    
    # Use Ticket Quantity as the sales metric
    if 'Ticket Quantity' in ticket_df.columns:
        sales_col = 'Ticket Quantity'
    elif 'ticket_quantity' in ticket_df.columns:
        sales_col = 'ticket_quantity'
    else:
        # Look for quantity-related columns
        qty_cols = [col for col in ticket_df.columns if 'quantity' in col.lower()]
        if qty_cols:
            sales_col = qty_cols[0]
        else:
            # Count ticket records as sales
            ticket_df['ticket_count'] = 1
            sales_col = 'ticket_count'
    
    if sales_col is None:
        print("❌ No sales column found in ticket data")
        return None
    
    print(f"  Using sales column: {sales_col}")
    
    # Standardize date formats
    marketing_df['date_id'] = pd.to_datetime(marketing_df['date_id'])
    ticket_df[date_col] = pd.to_datetime(ticket_df[date_col])
    
    # Aggregate ticket sales by date if needed
    ticket_agg = ticket_df.groupby(date_col)[sales_col].sum().reset_index()
    ticket_agg.columns = ['date_id', 'ticket_sales']
    
    print(f"✓ Ticket sales aggregated by date: {ticket_agg.shape}")
    print(f"  Total tickets: {ticket_agg['ticket_sales'].sum():,}")
    print(f"  Date range: {ticket_agg['date_id'].min()} to {ticket_agg['date_id'].max()}")
    
    # Merge data
    merged_df = pd.merge(marketing_df, ticket_agg, on='date_id', how='left')
    
    # Fill missing ticket sales with 0
    merged_df['ticket_sales'] = merged_df['ticket_sales'].fillna(0)
    
    print(f"✓ Data merged: {merged_df.shape}")
    print(f"  Non-zero ticket sales periods: {(merged_df['ticket_sales'] > 0).sum()}")
    
    # Add revenue per ticket (estimate)
    avg_ticket_price = 500  # €500 estimate - adjust as needed
    merged_df['revenue_per_ticket'] = avg_ticket_price
    
    # Replace clicks with ticket_sales as the KPI
    merged_df['clicks'] = merged_df['ticket_sales']  # Keep original column name for compatibility
    
    # Save merged data
    output_path = '/Users/aaronmeagher/Work/google_meridian/google/ws_attempt/data/raw_data/merged_marketing_tickets.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f"✓ Merged data saved: {output_path}")
    
    # Summary statistics
    print(f"\n📊 MERGED DATA SUMMARY:")
    print(f"  Total periods: {len(merged_df)}")
    print(f"  Total ticket sales: {merged_df['ticket_sales'].sum():,}")
    print(f"  Total marketing spend: €{merged_df[['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend']].sum().sum():,}")
    print(f"  Periods with tickets: {(merged_df['ticket_sales'] > 0).sum()}")
    print(f"  Periods with marketing: {(merged_df[['google_spend', 'facebook_spend', 'linkedin_spend', 'reddit_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend']].sum(axis=1) > 0).sum()}")
    
    # Show sample of merged data
    print(f"\n🔍 SAMPLE DATA:")
    sample = merged_df[['date_id', 'ticket_sales', 'google_spend', 'facebook_spend']].head()
    print(sample.to_string(index=False))
    
    return output_path

if __name__ == "__main__":
    merge_data()