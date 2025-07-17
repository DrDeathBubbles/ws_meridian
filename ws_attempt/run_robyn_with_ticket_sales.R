#!/usr/bin/env Rscript

# Load required packages
library(Robyn)
library(dplyr)
library(lubridate)
library(ggplot2)

# Load the InputCollect object prepared by Python
load("robyn_input_ticket_sales.RData")

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
  plot_folder = "plots_ticket_sales"
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

# Save results
save(OutputModels, OutputCollect, ExportedModel, BudgetAllocator, 
     file = "robyn_results_ticket_sales.RData")