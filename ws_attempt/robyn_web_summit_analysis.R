library(Robyn)
library(dplyr)
library(lubridate)

# Load and prepare data
data <- read.csv("data/raw_data/regularized_web_summit_data_fixed.csv")

# Data preprocessing
data$date_id <- as.Date(data$date_id)
data <- data %>%
  arrange(date_id) %>%
  filter(!is.na(date_id))

# Define dependent variable (KPI)
dep_var <- "conference_year"

# Define media channels
paid_media_spends <- c(
  "reddit_spend", "linkedin_spend", "facebook_spend", 
  "google_spend", "bing_spend", "tiktok_spend", 
  "twitter_spend", "instagram_spend"
)

# Define media impressions/exposures
paid_media_vars <- c(
  "reddit_impressions", "linkedin_impressions", "facebook_impressions",
  "google_impressions", "bing_impressions", "tiktok_impressions", 
  "twitter_impressions", "instagram_impressions"
)

# Context variables
context_vars <- c("days_to_event", "weeks_to_event")

# Set up Robyn input
InputCollect <- robyn_inputs(
  dt_input = data,
  dt_holidays = dt_prophet_holidays,
  date_var = "date_id",
  dep_var = dep_var,
  dep_var_type = "conversion",
  prophet_vars = c("trend", "season", "holiday"),
  prophet_country = "US",
  context_vars = context_vars,
  paid_media_spends = paid_media_spends,
  paid_media_vars = paid_media_vars,
  window_start = min(data$date_id),
  window_end = max(data$date_id),
  adstock = "geometric"
)

# Hyperparameters
hyper_names <- c(
  paste0(paid_media_spends, "_alphas"),
  paste0(paid_media_spends, "_gammas"),
  paste0(paid_media_spends, "_thetas")
)

# Set hyperparameter bounds
hyperparameters <- list(
  reddit_spend_alphas = c(0.5, 3),
  reddit_spend_gammas = c(0.3, 1),
  reddit_spend_thetas = c(0, 0.3),
  
  linkedin_spend_alphas = c(0.5, 3),
  linkedin_spend_gammas = c(0.3, 1),
  linkedin_spend_thetas = c(0, 0.3),
  
  facebook_spend_alphas = c(0.5, 3),
  facebook_spend_gammas = c(0.3, 1),
  facebook_spend_thetas = c(0, 0.3),
  
  google_spend_alphas = c(0.5, 3),
  google_spend_gammas = c(0.3, 1),
  google_spend_thetas = c(0, 0.3),
  
  bing_spend_alphas = c(0.5, 3),
  bing_spend_gammas = c(0.3, 1),
  bing_spend_thetas = c(0, 0.3),
  
  tiktok_spend_alphas = c(0.5, 3),
  tiktok_spend_gammas = c(0.3, 1),
  tiktok_spend_thetas = c(0, 0.3),
  
  twitter_spend_alphas = c(0.5, 3),
  twitter_spend_gammas = c(0.3, 1),
  twitter_spend_thetas = c(0, 0.3),
  
  instagram_spend_alphas = c(0.5, 3),
  instagram_spend_gammas = c(0.3, 1),
  instagram_spend_thetas = c(0, 0.3)
)

InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)

# Run Robyn model
OutputModels <- robyn_run(
  InputCollect = InputCollect,
  cores = 4,
  iterations = 2000,
  trials = 5,
  outputs = FALSE
)

# Select best model
OutputCollect <- robyn_outputs(
  InputCollect = InputCollect,
  OutputModels = OutputModels,
  csv_out = "pareto",
  clusters = TRUE,
  plot_pareto = TRUE,
  plot_folder = "plots"
)

# Get model results
select_model <- OutputCollect$allSolutions[1]
ExportedModel <- robyn_write(InputCollect, OutputCollect, select_model)

# Budget allocation
BudgetAllocator <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_historical_response",
  channel_constr_low = 0.7,
  channel_constr_up = 1.2
)

print("Robyn analysis complete!")
print(paste("Best model:", select_model))
print("Budget allocation results:")
print(BudgetAllocator$dt_optimOut)