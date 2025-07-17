import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Load the data
data_path = "data/raw_data/merged_marketing_tickets_fixed.csv"
data = pd.read_csv(data_path)
data['date_id'] = pd.to_datetime(data['date_id'])

# Define marketing channels
marketing_channels = [
    'reddit_spend', 'linkedin_spend', 'facebook_spend', 
    'google_spend', 'bing_spend', 'tiktok_spend', 
    'twitter_spend', 'instagram_spend'
]

# Build regression model
X = data[marketing_channels]
y = data['ticket_sales']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for test set
y_pred_test = model.predict(X_test)

# Make predictions for all data
y_pred_all = model.predict(X)

# 1. Create actual vs predicted plot for test data
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Ticket Sales')
plt.ylabel('Predicted Ticket Sales')
plt.title('Actual vs Predicted Ticket Sales (Test Set)')
plt.grid(True, alpha=0.3)
plt.savefig('robyn/robyn_output/actual_vs_predicted.png')

# 2. Create time series of actual vs predicted
data_with_pred = data.copy()
data_with_pred['predicted_tickets'] = y_pred_all

plt.figure(figsize=(15, 8))
plt.plot(data_with_pred['date_id'], data_with_pred['ticket_sales'], label='Actual Ticket Sales', alpha=0.7)
plt.plot(data_with_pred['date_id'], data_with_pred['predicted_tickets'], label='Predicted Ticket Sales', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Ticket Sales')
plt.title('Actual vs Predicted Ticket Sales Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('robyn/robyn_output/actual_vs_predicted_time_series.png')

# 3. Create response curves for each channel
fig, axs = plt.subplots(2, 4, figsize=(20, 12))
axs = axs.flatten()

for i, channel in enumerate(marketing_channels):
    # Get current spend for this channel
    current_spend = data[channel].values
    
    # Create range of spend values from 0 to 2x max spend
    max_spend = np.max(current_spend) * 2
    spend_range = np.linspace(0, max_spend, 100)
    
    # Create prediction data with only this channel varying
    X_pred = np.zeros((100, len(marketing_channels)))
    X_pred[:, i] = spend_range
    
    # Predict ticket sales
    y_pred = model.predict(X_pred)
    
    # Plot response curve
    axs[i].plot(spend_range, y_pred)
    axs[i].set_title(f'Response Curve: {channel}')
    axs[i].set_xlabel('Marketing Spend (€)')
    axs[i].set_ylabel('Predicted Ticket Sales')
    axs[i].grid(True, alpha=0.3)
    
    # Add current average spend as vertical line
    avg_spend = np.mean(current_spend[current_spend > 0]) if np.sum(current_spend > 0) > 0 else 0
    if avg_spend > 0:
        axs[i].axvline(x=avg_spend, color='r', linestyle='--', label='Avg Spend')
        axs[i].legend()

plt.tight_layout()
plt.savefig('robyn/robyn_output/response_curves.png')

# 4. Create normalized response curves (easier to compare)
fig, axs = plt.subplots(2, 4, figsize=(20, 12))
axs = axs.flatten()

for i, channel in enumerate(marketing_channels):
    # Get current spend for this channel
    current_spend = data[channel].values
    
    # Create range of spend values from 0 to 2x max spend
    max_spend = np.max(current_spend) * 2
    if max_spend > 0:
        spend_range = np.linspace(0, max_spend, 100)
        
        # Create prediction data with only this channel varying
        X_pred = np.zeros((100, len(marketing_channels)))
        X_pred[:, i] = spend_range
        
        # Predict ticket sales
        y_pred = model.predict(X_pred)
        
        # Normalize spend (as % of max) and response
        norm_spend = spend_range / max_spend * 100
        max_response = np.max(y_pred) if np.max(y_pred) > 0 else 1
        norm_response = y_pred / max_response * 100
        
        # Plot normalized response curve
        axs[i].plot(norm_spend, norm_response)
        axs[i].set_title(f'Normalized Response: {channel}')
        axs[i].set_xlabel('Marketing Spend (% of max)')
        axs[i].set_ylabel('Response (% of max)')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlim([0, 100])
        axs[i].set_ylim([0, 100])
        
        # Add current average spend as vertical line
        avg_spend = np.mean(current_spend[current_spend > 0]) if np.sum(current_spend > 0) > 0 else 0
        if avg_spend > 0:
            norm_avg = avg_spend / max_spend * 100
            axs[i].axvline(x=norm_avg, color='r', linestyle='--', label='Avg Spend')
            axs[i].legend()
    else:
        axs[i].text(0.5, 0.5, 'No spend data', ha='center', va='center', transform=axs[i].transAxes)

plt.tight_layout()
plt.savefig('robyn/robyn_output/normalized_response_curves.png')

print("Additional plots created and saved to robyn/robyn_output/")