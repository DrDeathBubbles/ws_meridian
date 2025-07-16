#!/usr/bin/env Rscript

# Install required packages if not already installed
if (!require("Robyn")) {
  if (!require("remotes")) install.packages("remotes")
  remotes::install_github("facebookexperimental/Robyn/R")
}

required_packages <- c("dplyr", "lubridate", "ggplot2", "reticulate")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Set working directory
setwd("/Users/aaronmeagher/Work/google_meridian/google/ws_attempt")

# Source the main analysis script
source("robyn_web_summit_analysis.R")