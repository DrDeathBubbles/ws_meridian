#!/bin/bash

# Process ticket sales data and prepare Robyn input
echo "Processing ticket sales data and preparing Robyn input..."
python3 robyn_with_ticket_sales.py

# Run Robyn model
echo "Running Robyn model with ticket sales as KPI..."
Rscript run_robyn_with_ticket_sales.R

echo "Analysis complete. Results saved in robyn_results_ticket_sales.RData"