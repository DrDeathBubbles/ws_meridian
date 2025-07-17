import pandas as pd
import numpy as np
import os

# Process ticket sales data if not already done
if not os.path.exists('data_ws/daily_ticket_sales.csv'):
    print("Processing ticket sales data...")
    exec(open('data_ws/process_ticket_sales.py').read())
else:
    print("Using existing processed ticket sales data")

# Load and prepare data
data = pd.read_csv("data/raw_data/regularized_web_summit_data_fixed.csv")
data['date_id'] = pd.to_datetime(data['date_id'])
data = data.sort_values('date_id').dropna(subset=['date_id'])

# Load ticket sales data
ticket_sales = pd.read_csv("data_ws/daily_ticket_sales.csv")
ticket_sales['date_id'] = pd.to_datetime(ticket_sales['date_id'])

# Merge with main data
data = pd.merge(data, ticket_sales, on='date_id', how='left')

# Fill NA values with 0 for ticket_sales
data['ticket_sales'] = data['ticket_sales'].fillna(0)

# Save the merged data
data.to_csv("data/raw_data/web_summit_data_with_ticket_sales.csv", index=False)

print("Data prepared and saved to data/raw_data/web_summit_data_with_ticket_sales.csv")