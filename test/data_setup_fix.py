#!/usr/bin/env python3
"""
Fix for coord_to_columns media data requirement
"""

from meridian.data import load

def create_coord_to_columns_with_media():
    """Create coord_to_columns with required media fields"""
    return load.CoordToColumns(
        time='date_id',
        geo='geo', 
        kpi='conversions',
        # REQUIRED: Either media OR reach/frequency data
        media=['google_impressions', 'facebook_impressions'],  # Your impression columns
        media_spend=['google_spend', 'facebook_spend'],       # Your spend columns
        # Optional fields
        population='population',  # If you have population data
        controls=['control1', 'control2'],  # If you have control variables
    )

def create_coord_to_columns_with_rf():
    """Alternative: Use reach/frequency instead of media"""
    return load.CoordToColumns(
        time='date_id',
        geo='geo',
        kpi='conversions', 
        # REQUIRED: Reach/frequency data instead of media
        reach=['google_reach', 'facebook_reach'],
        frequency=['google_frequency', 'facebook_frequency'],
        rf_spend=['google_spend', 'facebook_spend'],
    )

# Quick fix for your current setup:
coord_to_columns = load.CoordToColumns(
    time='date_id',
    geo='geo',
    kpi='conversions',
    media_spend=['google_spend', 'facebook_spend', 'linkedin_spend'],  # Add your spend columns
    media=['google_impressions', 'facebook_impressions', 'linkedin_impressions']  # Add impression columns
)