import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('output', exist_ok=True)

print("Combining ticket sales and revenue data...")

# Load ticket sales data
ticket_data = pd.read_csv('data_ws/TicketSales.csv')
ticket_data['Transaction_Date'] = pd.to_datetime(ticket_data['Transaction Date'], errors='coerce')

# Aggregate ticket sales by date
daily_ticket_counts = (ticket_data
    .dropna(subset=['Transaction_Date'])
    .groupby(ticket_data['Transaction_Date'].dt.date)
    .size()
    .reset_index(name='ticket_count')
    .rename(columns={'Transaction_Date': 'Date'})
)
daily_ticket_counts['Date'] = pd.to_datetime(daily_ticket_counts['Date'])

# Load revenue data
revenue_data = pd.read_csv('data_ws/ga_revenue_over_time.csv')
revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])

# Clean revenue data (replace negative values with 0)
revenue_data['Gross Paid Eur'] = revenue_data['Gross Paid Eur'].apply(lambda x: max(0, x))

# Merge ticket counts and revenue data
combined_data = pd.merge(daily_ticket_counts, revenue_data, on='Date', how='outer')

# Fill missing values
combined_data['ticket_count'] = combined_data['ticket_count'].fillna(0)
combined_data['Gross Paid Eur'] = combined_data['Gross Paid Eur'].fillna(0)

# Calculate average revenue per ticket
combined_data['avg_revenue_per_ticket'] = np.where(
    combined_data['ticket_count'] > 0,
    combined_data['Gross Paid Eur'] / combined_data['ticket_count'],
    0
)

# Sort by date
combined_data = combined_data.sort_values('Date')

# Save combined data
combined_data.to_csv('output/combined_ticket_revenue.csv', index=False)
print(f"Combined data saved to output/combined_ticket_revenue.csv")

# Load marketing data
try:
    marketing_data = pd.read_csv("data/raw_data/regularized_web_summit_data_fixed.csv")
    marketing_data['date_id'] = pd.to_datetime(marketing_data['date_id'])
    
    # Rename Date column for consistency
    combined_data = combined_data.rename(columns={'Date': 'date_id'})
    
    # Merge with marketing data
    merged_data = pd.merge(marketing_data, combined_data, on='date_id', how='left')
    
    # Fill missing values
    merged_data['ticket_count'] = merged_data['ticket_count'].fillna(0)
    merged_data['Gross Paid Eur'] = merged_data['Gross Paid Eur'].fillna(0)
    merged_data['avg_revenue_per_ticket'] = merged_data['avg_revenue_per_ticket'].fillna(0)
    
    # Save merged data
    merged_data.to_csv('output/marketing_ticket_revenue.csv', index=False)
    print(f"Marketing data merged with ticket and revenue data saved to output/marketing_ticket_revenue.csv")
    
    # Create a version with ticket count as the KPI (for compatibility with existing models)
    merged_data_tickets = merged_data.copy()
    merged_data_tickets['ticket_sales'] = merged_data_tickets['ticket_count']
    merged_data_tickets.to_csv('output/merged_marketing_tickets_updated.csv', index=False)
    print(f"Updated merged marketing and ticket data saved to output/merged_marketing_tickets_updated.csv")
    
    # Create a version with revenue as the KPI
    merged_data_revenue = merged_data.copy()
    merged_data_revenue['revenue'] = merged_data_revenue['Gross Paid Eur']
    merged_data_revenue.to_csv('output/merged_marketing_revenue.csv', index=False)
    print(f"Merged marketing and revenue data saved to output/merged_marketing_revenue.csv")
    
except Exception as e:
    print(f"Error processing marketing data: {e}")

# Plot ticket count vs revenue
plt.figure(figsize=(12, 6))
plt.scatter(combined_data['ticket_count'], combined_data['Gross Paid Eur'], alpha=0.6)
plt.title('Ticket Count vs Revenue')
plt.xlabel('Number of Tickets')
plt.ylabel('Revenue (EUR)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/ticket_count_vs_revenue.png')

# Plot time series of tickets and revenue
fig, ax1 = plt.subplots(figsize=(15, 6))

# Ticket count on left y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Ticket Count', color='tab:blue')
ax1.plot(combined_data['date_id'], combined_data['ticket_count'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Revenue on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Revenue (EUR)', color='tab:red')
ax2.plot(combined_data['date_id'], combined_data['Gross Paid Eur'], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Ticket Count and Revenue Over Time')
fig.tight_layout()
plt.savefig('output/ticket_revenue_time_series.png')

print("\nAnalysis complete! Results saved to the 'output' directory.")