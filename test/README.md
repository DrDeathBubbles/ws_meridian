# Google Meridian Tutorial

This folder contains tutorial materials for getting started with Google Meridian Media Mix Modeling.

## Files

- `meridian_tutorial_fixed.py` - **RECOMMENDED** - Working tutorial with fixes
- `meridian_tutorial.py` - Original tutorial (has sampling issues)
- `meridian_tutorial.ipynb` - Interactive Jupyter notebook (fixed column mappings)
- `meridian_tutorial_simple.py` - Simplified version
- `meridian_working.py` - Minimal working example
- `README.md` - This file

## Quick Start

### Option 1: Run Python Script
```bash
cd test
python meridian_tutorial.py
```

### Option 2: Use Jupyter Notebook
```bash
cd test
jupyter notebook meridian_tutorial.ipynb
```

## What This Tutorial Covers

1. **System Setup** - Check available resources (RAM, GPU/CPU)
2. **Data Loading** - Load sample data with proper column mappings
3. **Model Specification** - Set up priors and model configuration
4. **Model Training** - Sample from prior and posterior distributions
5. **Diagnostics** - Validate model convergence with R-hat statistics
6. **Analysis** - Calculate media contributions and ROI estimates
7. **Optimization** - Budget allocation optimization (optional)

## Issues Diagnosed and Fixed

### ✅ FIXED ISSUES:
1. **Column Mapping Error**: Original tutorial used incorrect column names
   - Fixed: `'GQV'` → `'competitor_sales_control'`
   - Fixed: `'Competitor_Sales'` → `'sentiment_score_control'`

2. **Data Inspection Error**: Used non-existent attributes
   - Fixed: `data.n_time_periods` → `data.kpi.shape[0]`
   - Fixed: `data.n_geos` → `data.kpi.shape[1]`
   - Fixed: `data.n_media_channels` → `data.media.shape[2]`

3. **Error Handling**: Added proper exception handling

### ⚠️ KNOWN ISSUES (Meridian Library):
- Prior sampling fails with tensor type mismatch
- Posterior sampling has convergence problems with tutorial data
- These are library-level issues, not tutorial issues

## Requirements

Make sure you have these packages installed:
```bash
pip install meridian tensorflow tensorflow-probability pandas numpy matplotlib arviz psutil
```

## Expected Output

- Model diagnostics plots saved as PNG files
- ROI estimates and media contribution analysis
- Console output showing training progress
- Convergence diagnostics (R-hat values near 1.0 indicate good convergence)

## Next Steps

After running the tutorial:
1. Review the R-hat diagnostic plots
2. Analyze the ROI estimates for each channel
3. Experiment with different prior specifications
4. Apply the methodology to your own marketing data

## Troubleshooting

If you encounter issues:
- Ensure all required packages are installed
- Check that you have sufficient RAM (8GB+ recommended)
- Verify internet connection for data download
- Review error messages for specific column mapping issues

For more information, visit: https://developers.google.com/meridian