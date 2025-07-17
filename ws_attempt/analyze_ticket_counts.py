import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

print("Processing ticket sales data...")

# Read the ticket sales data
ticket_data = pd.read_csv('data_ws/TicketSales.csv')

# Convert Transaction Date to datetime format
ticket_data['Transaction_Date'] = pd.to_datetime(ticket_data['Transaction Date'], errors='coerce')

# Extract event information
ticket_data['Event_Year'] = ticket_data['Event Name'].str.extract(r'(\d{4})').astype(float)

# Analyze ticket counts by event and product type
event_product_counts = ticket_data.groupby(['Event Name', 'Product Name']).size().reset_index(name='ticket_count')
event_product_counts = event_product_counts.sort_values('ticket_count', ascending=False)
event_product_counts.to_csv('output/event_product_counts.csv', index=False)
print(f"Event and product type counts saved to output/event_product_counts.csv")

# Analyze ticket counts by status
status_counts = ticket_data.groupby(['Order Status']).size().reset_index(name='ticket_count')
status_counts = status_counts.sort_values('ticket_count', ascending=False)
status_counts.to_csv('output/status_counts.csv', index=False)
print(f"Order status counts saved to output/status_counts.csv")

# Aggregate ticket counts by date
daily_ticket_counts = (ticket_data
    .dropna(subset=['Transaction_Date'])
    .groupby(ticket_data['Transaction_Date'].dt.date)
    .size()
    .reset_index(name='ticket_count')
    .rename(columns={'Transaction_Date': 'date_id'})
)

# Convert date_id to datetime for proper plotting
daily_ticket_counts['date_id'] = pd.to_datetime(daily_ticket_counts['date_id'])

# Save the aggregated data
daily_ticket_counts.to_csv('output/daily_ticket_counts.csv', index=False)
print(f"Daily ticket counts saved to output/daily_ticket_counts.csv")

# Plot ticket counts over time
plt.figure(figsize=(15, 7))
plt.plot(daily_ticket_counts['date_id'], daily_ticket_counts['ticket_count'])
plt.title('Daily Ticket Sales Count Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tickets Sold')
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ticket_counts_time_series.png')
print(f"Ticket counts time series plot saved to output/ticket_counts_time_series.png")

# Calculate total tickets by event
event_counts = ticket_data.groupby('Event Name').size().sort_values(ascending=False)
event_counts.to_csv('output/event_counts.csv')

# Plot tickets by event
plt.figure(figsize=(12, 8))
event_counts.plot(kind='bar')
plt.title('Total Tickets by Event')
plt.xlabel('Event')
plt.ylabel('Number of Tickets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/tickets_by_event.png')
print(f"Tickets by event plot saved to output/tickets_by_event.png")

# Load marketing data and merge with ticket counts
try:
    marketing_data = pd.read_csv("data/raw_data/regularized_web_summit_data_fixed.csv")
    marketing_data['date_id'] = pd.to_datetime(marketing_data['date_id'])
    
    # Merge with ticket counts data
    merged_data = pd.merge(marketing_data, daily_ticket_counts, on='date_id', how='left')
    
    # Fill NA values with 0 for ticket_count
    merged_data['ticket_count'] = merged_data['ticket_count'].fillna(0)
    
    # Save the merged data
    merged_data.to_csv("output/web_summit_data_with_ticket_counts.csv", index=False)
    print(f"Marketing data merged with ticket counts and saved to output/web_summit_data_with_ticket_counts.csv")
    
    # Analyze correlations between marketing channels and ticket counts
    print("\nAnalyzing correlations between marketing channels and ticket counts...")
    
    # Select relevant columns for correlation analysis
    correlation_columns = [
        'ticket_count', 'reddit_spend', 'linkedin_spend', 'facebook_spend', 
        'google_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend',
        'days_to_event', 'weeks_to_event'
    ]
    
    # Calculate correlations
    correlation_data = merged_data[correlation_columns].corr()
    
    # Save correlation data
    correlation_data.to_csv('output/marketing_ticket_counts_correlation.csv')
    print(f"Correlation analysis saved to output/marketing_ticket_counts_correlation.csv")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Marketing Channels and Ticket Counts')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap_counts.png')
    print(f"Correlation heatmap saved to output/correlation_heatmap_counts.png")
    
    # Simple regression analysis
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data for regression
    spend_columns = [col for col in merged_data.columns if col.endswith('_spend')]
    X = merged_data[spend_columns]
    y = merged_data['ticket_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Channel': spend_columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    # Save coefficients
    coefficients.to_csv('output/marketing_channel_coefficients_counts.csv', index=False)
    print(f"Regression coefficients saved to output/marketing_channel_coefficients_counts.csv")
    
    # Print model performance
    print(f"\nRegression Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Print top channels by impact
    print("\nTop Marketing Channels by Impact on Ticket Counts:")
    for i, row in coefficients.head(3).iterrows():
        print(f"{row['Channel']}: {row['Coefficient']:.4f}")
    
except Exception as e:
    print(f"Error processing marketing data: {e}")
    
print("\nAnalysis complete!")