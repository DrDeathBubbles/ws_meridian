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

# Aggregate ticket sales by date
daily_ticket_sales = (ticket_data
    .dropna(subset=['Transaction_Date'])
    .groupby(ticket_data['Transaction_Date'].dt.date)
    .size()
    .reset_index(name='ticket_sales')
    .rename(columns={'Transaction_Date': 'date_id'})
)

# Convert date_id to datetime for proper plotting
daily_ticket_sales['date_id'] = pd.to_datetime(daily_ticket_sales['date_id'])

# Save the aggregated data
daily_ticket_sales.to_csv('output/daily_ticket_sales.csv', index=False)
print(f"Ticket sales data processed and saved to output/daily_ticket_sales.csv")

# Load marketing data
try:
    marketing_data = pd.read_csv("data/raw_data/regularized_web_summit_data_fixed.csv")
    marketing_data['date_id'] = pd.to_datetime(marketing_data['date_id'])
    
    # Merge with ticket sales data
    merged_data = pd.merge(marketing_data, daily_ticket_sales, on='date_id', how='left')
    
    # Fill NA values with 0 for ticket_sales
    merged_data['ticket_sales'] = merged_data['ticket_sales'].fillna(0)
    
    # Save the merged data
    merged_data.to_csv("output/web_summit_data_with_ticket_sales.csv", index=False)
    print(f"Marketing data merged with ticket sales and saved to output/web_summit_data_with_ticket_sales.csv")
    
    # Analyze correlations between marketing channels and ticket sales
    print("\nAnalyzing correlations between marketing channels and ticket sales...")
    
    # Select relevant columns for correlation analysis
    correlation_columns = [
        'ticket_sales', 'reddit_spend', 'linkedin_spend', 'facebook_spend', 
        'google_spend', 'bing_spend', 'tiktok_spend', 'twitter_spend', 'instagram_spend',
        'days_to_event', 'weeks_to_event'
    ]
    
    # Calculate correlations
    correlation_data = merged_data[correlation_columns].corr()
    
    # Save correlation data
    correlation_data.to_csv('output/marketing_ticket_sales_correlation.csv')
    print(f"Correlation analysis saved to output/marketing_ticket_sales_correlation.csv")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Marketing Channels and Ticket Sales')
    plt.tight_layout()
    plt.savefig('output/correlation_heatmap.png')
    print(f"Correlation heatmap saved to output/correlation_heatmap.png")
    
    # Plot ticket sales over time
    plt.figure(figsize=(15, 7))
    plt.plot(daily_ticket_sales['date_id'], daily_ticket_sales['ticket_sales'])
    plt.title('Daily Ticket Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tickets Sold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/ticket_sales_time_series.png')
    print(f"Ticket sales time series plot saved to output/ticket_sales_time_series.png")
    
    # Calculate total marketing spend by channel
    spend_columns = [col for col in merged_data.columns if col.endswith('_spend')]
    total_spend = merged_data[spend_columns].sum().sort_values(ascending=False)
    
    # Plot marketing spend by channel
    plt.figure(figsize=(12, 6))
    total_spend.plot(kind='bar')
    plt.title('Total Marketing Spend by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Total Spend (EUR)')
    plt.tight_layout()
    plt.savefig('output/marketing_spend_by_channel.png')
    print(f"Marketing spend by channel plot saved to output/marketing_spend_by_channel.png")
    
    # Simple regression analysis
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data for regression
    X = merged_data[spend_columns]
    y = merged_data['ticket_sales']
    
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
    coefficients.to_csv('output/marketing_channel_coefficients.csv', index=False)
    print(f"Regression coefficients saved to output/marketing_channel_coefficients.csv")
    
    # Plot coefficients
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Channel', y='Coefficient', data=coefficients)
    plt.title('Marketing Channel Impact on Ticket Sales')
    plt.xlabel('Channel')
    plt.ylabel('Coefficient (Impact)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/channel_impact.png')
    print(f"Channel impact plot saved to output/channel_impact.png")
    
    # Print model performance
    print(f"\nRegression Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Print top channels by impact
    print("\nTop Marketing Channels by Impact on Ticket Sales:")
    for i, row in coefficients.head(3).iterrows():
        print(f"{row['Channel']}: {row['Coefficient']:.4f}")
    
except Exception as e:
    print(f"Error processing marketing data: {e}")
    
print("\nAnalysis complete!")