import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Load the merged marketing and ticket sales data
data_path = "data/raw_data/merged_marketing_tickets_fixed.csv"
data = pd.read_csv(data_path)
data['date_id'] = pd.to_datetime(data['date_id'])

print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Date range: {data['date_id'].min()} to {data['date_id'].max()}")
print(f"Total ticket sales: {data['ticket_sales'].sum()}")

# Analyze marketing spend vs ticket sales
marketing_channels = [
    'reddit_spend', 'linkedin_spend', 'facebook_spend', 
    'google_spend', 'bing_spend', 'tiktok_spend', 
    'twitter_spend', 'instagram_spend'
]

# Calculate correlation between marketing spend and ticket sales
correlation = data[marketing_channels + ['ticket_sales']].corr()
print("\nCorrelation between marketing channels and ticket sales:")
print(correlation['ticket_sales'].sort_values(ascending=False))

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation between Marketing Channels and Ticket Sales')
plt.tight_layout()
plt.savefig('output/marketing_ticket_correlation.png')

# Build regression model
X = data[marketing_channels]
y = data['ticket_sales']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Get channel coefficients
coefficients = pd.DataFrame({
    'Channel': marketing_channels,
    'Coefficient': model.coef_
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

print("\nMarketing Channel Impact on Ticket Sales:")
for i, row in coefficients.iterrows():
    print(f"{row['Channel']}: {row['Coefficient']:.4f}")

# Plot coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Channel', data=coefficients)
plt.title('Marketing Channel Impact on Ticket Sales')
plt.tight_layout()
plt.savefig('output/channel_impact.png')

# Calculate ROI (tickets per € spent)
total_spend = data[marketing_channels].sum()
channel_spend = total_spend[total_spend > 0]

# Calculate predicted tickets for each channel
channel_contribution = {}
for channel in marketing_channels:
    if total_spend[channel] > 0:
        # Create a dataframe with only this channel's spend
        X_channel = np.zeros_like(X)
        channel_idx = marketing_channels.index(channel)
        X_channel[:, channel_idx] = X.iloc[:, channel_idx]
        
        # Predict tickets from this channel
        y_channel = model.predict(X_channel)
        channel_contribution[channel] = y_channel.sum()

# Calculate ROI
roi = {}
for channel, contribution in channel_contribution.items():
    roi[channel] = contribution / total_spend[channel]

roi_df = pd.DataFrame({
    'Channel': list(roi.keys()),
    'ROI (Tickets per €)': list(roi.values())
}).sort_values('ROI (Tickets per €)', ascending=False)

print("\nROI Analysis (Tickets per €):")
for i, row in roi_df.iterrows():
    print(f"{row['Channel']}: {row['ROI (Tickets per €)']:.4f}")

# Plot ROI
plt.figure(figsize=(10, 6))
sns.barplot(x='ROI (Tickets per €)', y='Channel', data=roi_df)
plt.title('ROI by Marketing Channel (Tickets per €)')
plt.tight_layout()
plt.savefig('output/channel_roi.png')

# Plot ticket sales over time
plt.figure(figsize=(15, 6))
plt.plot(data['date_id'], data['ticket_sales'])
plt.title('Ticket Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tickets')
plt.grid(True)
plt.tight_layout()
plt.savefig('output/ticket_sales_time_series.png')

# Save results to CSV
coefficients.to_csv('output/channel_coefficients.csv', index=False)
roi_df.to_csv('output/channel_roi.csv', index=False)

print("\nAnalysis complete! Results saved to the 'output' directory.")