import pandas as pd
import numpy as np
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Activate automatic conversion between pandas and R dataframes
pandas2ri.activate()

# Import R packages
base = importr('base')
robyn = importr('Robyn')
dplyr = importr('dplyr')
lubridate = importr('lubridate')

# Process ticket sales data if not already done
if not os.path.exists('data_ws/daily_ticket_sales.csv'):
    print("Processing ticket sales data...")
    exec(open('data_ws/process_ticket_sales.py').read())

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

# Convert to R dataframe
r_data = pandas2ri.py2rpy(data)

# Define variables for Robyn
dep_var = "ticket_sales"
paid_media_spends = [
    "reddit_spend", "linkedin_spend", "facebook_spend", 
    "google_spend", "bing_spend", "tiktok_spend", 
    "twitter_spend", "instagram_spend"
]
paid_media_vars = [
    "reddit_impressions", "linkedin_impressions", "facebook_impressions",
    "google_impressions", "bing_impressions", "tiktok_impressions", 
    "twitter_impressions", "instagram_impressions"
]
context_vars = ["days_to_event", "weeks_to_event"]

# Convert lists to R vectors
r_paid_media_spends = robjects.StrVector(paid_media_spends)
r_paid_media_vars = robjects.StrVector(paid_media_vars)
r_context_vars = robjects.StrVector(context_vars)

# Get R's dt_prophet_holidays
dt_prophet_holidays = robyn.dt_prophet_holidays

# Set up Robyn input
InputCollect = robyn.robyn_inputs(
    dt_input=r_data,
    dt_holidays=dt_prophet_holidays,
    date_var="date_id",
    dep_var=dep_var,
    dep_var_type="conversion",
    prophet_vars=robjects.StrVector(["trend", "season", "holiday"]),
    prophet_country="US",
    context_vars=r_context_vars,
    paid_media_spends=r_paid_media_spends,
    paid_media_vars=r_paid_media_vars,
    window_start=robjects.r('min')(data['date_id']),
    window_end=robjects.r('max')(data['date_id']),
    adstock="geometric"
)

# Create hyperparameters dictionary
hyperparameters = {}
for media in paid_media_spends:
    hyperparameters[f"{media}_alphas"] = robjects.FloatVector([0.5, 3])
    hyperparameters[f"{media}_gammas"] = robjects.FloatVector([0.3, 1])
    hyperparameters[f"{media}_thetas"] = robjects.FloatVector([0, 0.3])

# Convert to R list
r_hyperparameters = robjects.ListVector(hyperparameters)

# Update InputCollect with hyperparameters
InputCollect = robyn.robyn_inputs(InputCollect=InputCollect, hyperparameters=r_hyperparameters)

print("Robyn model setup complete with ticket_sales as KPI")
print("Ready to run with: robyn_run(InputCollect)")

# Save the InputCollect object for use in R
robjects.r.assign("InputCollect", InputCollect)
robjects.r('save(InputCollect, file="robyn_input_ticket_sales.RData")')

print("Saved InputCollect to robyn_input_ticket_sales.RData")
print("You can now run the model in R with:")
print('load("robyn_input_ticket_sales.RData")')
print('OutputModels <- robyn_run(InputCollect = InputCollect, cores = 4, iterations = 2000, trials = 5)')