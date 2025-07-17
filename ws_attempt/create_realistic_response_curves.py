import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import os

# Load the data
data_path = "data/raw_data/merged_marketing_tickets_fixed.csv"
data = pd.read_csv(data_path)

# Define marketing channels
marketing_channels = [
    'reddit_spend', 'linkedin_spend', 'facebook_spend', 
    'google_spend', 'bing_spend', 'tiktok_spend', 
    'twitter_spend', 'instagram_spend'
]

# Create response curves with diminishing returns
fig, axs = plt.subplots(2, 4, figsize=(20, 12))
axs = axs.flatten()

for i, channel in enumerate(marketing_channels):
    # Get current spend for this channel
    current_spend = data[channel].values
    
    # Skip if no spend
    if np.sum(current_spend) == 0:
        axs[i].text(0.5, 0.5, 'No spend data', ha='center', va='center', transform=axs[i].transAxes)
        continue
    
    # Get ticket sales
    ticket_sales = data['ticket_sales'].values
    
    # Create polynomial model (degree 2 for diminishing returns)
    X = current_spend.reshape(-1, 1)
    X = X[X[:, 0] > 0]  # Only use non-zero spend
    if len(X) < 10:  # Skip if too few data points
        axs[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axs[i].transAxes)
        continue
        
    y = ticket_sales[current_spend > 0]
    
    # Create polynomial model
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    
    # Create range of spend values from 0 to 2x max spend
    max_spend = np.max(current_spend) * 2
    spend_range = np.linspace(0, max_spend, 100).reshape(-1, 1)
    
    # Predict ticket sales
    y_pred = model.predict(spend_range)
    
    # Ensure curve starts at origin
    y_pred = y_pred - y_pred[0] if y_pred[0] < 0 else y_pred
    
    # Apply square root transformation for more realistic diminishing returns
    # if channel has enough data points
    if len(X) >= 20:
        # Use square root model instead
        coef = np.sum(y) / np.sum(np.sqrt(X))
        y_pred_sqrt = coef * np.sqrt(spend_range)
        
        # Plot both models
        axs[i].plot(spend_range, y_pred, 'b-', alpha=0.5, label='Polynomial')
        axs[i].plot(spend_range, y_pred_sqrt, 'g-', label='Square Root')
        y_pred = y_pred_sqrt  # Use square root for further calculations
    else:
        # Just plot polynomial
        axs[i].plot(spend_range, y_pred, 'b-', label='Polynomial')
    
    # Plot actual data points
    axs[i].scatter(X, y, color='red', alpha=0.3, s=10, label='Actual Data')
    
    # Add titles and labels
    axs[i].set_title(f'Response Curve: {channel}')
    axs[i].set_xlabel('Marketing Spend (€)')
    axs[i].set_ylabel('Predicted Ticket Sales')
    axs[i].grid(True, alpha=0.3)
    
    # Add current average spend as vertical line
    avg_spend = np.mean(current_spend[current_spend > 0])
    axs[i].axvline(x=avg_spend, color='r', linestyle='--', label='Avg Spend')
    
    # Add legend
    axs[i].legend()

plt.tight_layout()
plt.savefig('robyn/robyn_output/realistic_response_curves.png')

# Create normalized response curves
fig, axs = plt.subplots(2, 4, figsize=(20, 12))
axs = axs.flatten()

for i, channel in enumerate(marketing_channels):
    # Get current spend for this channel
    current_spend = data[channel].values
    
    # Skip if no spend
    if np.sum(current_spend) == 0:
        axs[i].text(0.5, 0.5, 'No spend data', ha='center', va='center', transform=axs[i].transAxes)
        continue
    
    # Get ticket sales
    ticket_sales = data['ticket_sales'].values
    
    # Create square root model for diminishing returns
    X = current_spend.reshape(-1, 1)
    X = X[X[:, 0] > 0]  # Only use non-zero spend
    if len(X) < 10:  # Skip if too few data points
        axs[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axs[i].transAxes)
        continue
        
    y = ticket_sales[current_spend > 0]
    
    # Calculate coefficient for square root model
    coef = np.sum(y) / np.sum(np.sqrt(X))
    
    # Create range of spend values from 0 to 2x max spend
    max_spend = np.max(current_spend) * 2
    spend_range = np.linspace(0, max_spend, 100).reshape(-1, 1)
    
    # Predict ticket sales using square root model
    y_pred = coef * np.sqrt(spend_range)
    
    # Normalize spend and response
    norm_spend = spend_range / max_spend * 100
    max_response = np.max(y_pred)
    norm_response = y_pred / max_response * 100
    
    # Plot normalized response curve
    axs[i].plot(norm_spend, norm_response)
    
    # Add reference line for linear response
    axs[i].plot([0, 100], [0, 100], 'r--', alpha=0.3, label='Linear')
    
    # Add titles and labels
    axs[i].set_title(f'Normalized Response: {channel}')
    axs[i].set_xlabel('Marketing Spend (% of max)')
    axs[i].set_ylabel('Response (% of max)')
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xlim([0, 100])
    axs[i].set_ylim([0, 100])
    
    # Add current average spend as vertical line
    avg_spend = np.mean(current_spend[current_spend > 0])
    norm_avg = avg_spend / max_spend * 100
    axs[i].axvline(x=norm_avg, color='r', linestyle='--', label='Avg Spend')
    
    # Add legend
    axs[i].legend()

plt.tight_layout()
plt.savefig('robyn/robyn_output/realistic_normalized_curves.png')

# Create a summary plot showing all channels on one graph
plt.figure(figsize=(12, 8))

for i, channel in enumerate(marketing_channels):
    # Get current spend for this channel
    current_spend = data[channel].values
    
    # Skip if no spend
    if np.sum(current_spend) == 0:
        continue
    
    # Get ticket sales
    ticket_sales = data['ticket_sales'].values
    
    # Create square root model for diminishing returns
    X = current_spend.reshape(-1, 1)
    X = X[X[:, 0] > 0]  # Only use non-zero spend
    if len(X) < 10:  # Skip if too few data points
        continue
        
    y = ticket_sales[current_spend > 0]
    
    # Calculate coefficient for square root model
    coef = np.sum(y) / np.sum(np.sqrt(X))
    
    # Create range of spend values from 0 to 2x max spend
    max_spend = np.max(current_spend) * 2
    spend_range = np.linspace(0, max_spend, 100).reshape(-1, 1)
    
    # Predict ticket sales using square root model
    y_pred = coef * np.sqrt(spend_range)
    
    # Normalize spend and response
    norm_spend = spend_range / max_spend * 100
    max_response = np.max(y_pred)
    norm_response = y_pred / max_response * 100
    
    # Plot normalized response curve
    plt.plot(norm_spend, norm_response, label=channel)

# Add reference line for linear response
plt.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Linear')

# Add titles and labels
plt.title('Normalized Response Curves (All Channels)')
plt.xlabel('Marketing Spend (% of max)')
plt.ylabel('Response (% of max)')
plt.grid(True, alpha=0.3)
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.legend()
plt.tight_layout()
plt.savefig('robyn/robyn_output/all_channels_response.png')

print("Realistic response curves created and saved to robyn/robyn_output/")