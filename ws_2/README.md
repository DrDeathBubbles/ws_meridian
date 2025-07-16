# Web Summit Media Mix Modeling (ws_2)

This directory contains the latest version of the Google Meridian Media Mix Modeling implementation for Web Summit data.

## Files

- `improved_model_implementation.py` - Main implementation file with improved model configuration
- `improved_model.pkl` - Trained model with optimized parameters
- `regularized_web_summit_data_fixed.csv` - Latest processed data file
- `simple_model_plot.py` - Script for generating model prediction plots
- `model_prediction_plot.py` - Script for detailed prediction visualization
- `extract_model_parameters.py` - Utility for extracting model parameters
- `channel_attribution.py` - Analysis script for channel attribution

## Key Improvements

This version includes several improvements over previous implementations:

1. **Adjusted ROI priors** to better match the data scale
2. **Longer half-life values** (~7 days) for smoother adstock decay
3. **Proper sorting and smoothing** for visualizations
4. **Increased MCMC samples** for better convergence

## Usage

To run the model:

```bash
python improved_model_implementation.py
```

To visualize model predictions:

```bash
python simple_model_plot.py
```

For channel attribution analysis:

```bash
python channel_attribution.py
```