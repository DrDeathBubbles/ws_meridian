#!/usr/bin/env python3
"""
Robyn MMM Extensions - Advanced Analysis Features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

class RobynExtensions:
    def __init__(self, robyn_model):
        self.robyn = robyn_model
        
    def budget_optimization(self, total_budget, constraints=None):
        """Optimize budget allocation across channels"""
        print("💰 Running budget optimization...")
        
        # Calculate ROI if not already done
        if 'roi' not in self.robyn.model_results:
            self.robyn.calculate_roi()
        
        roi_data = self.robyn.model_results['roi']
        channels = list(roi_data.keys())
        
        # Simple optimization: allocate based on ROI efficiency
        roi_values = [roi_data[ch]['roi'] for ch in channels if roi_data[ch]['roi'] > 0]
        positive_channels = [ch for ch in channels if roi_data[ch]['roi'] > 0]
        
        if not positive_channels:
            return {}
        
        # Normalize ROI for allocation
        roi_weights = np.array(roi_values) / sum(roi_values)
        allocations = dict(zip(positive_channels, roi_weights * total_budget))
        
        return allocations
    
    def scenario_analysis(self, scenarios):
        """Run what-if scenarios"""
        print("🔮 Running scenario analysis...")
        
        results = {}
        for scenario_name, spend_changes in scenarios.items():
            # Apply spend changes and predict impact
            modified_data = self.robyn.data.copy()
            
            for channel, multiplier in spend_changes.items():
                spend_col = f"{channel.lower()}_spend"
                if spend_col in modified_data.columns:
                    modified_data[spend_col] *= multiplier
            
            # Recalculate predictions (simplified)
            results[scenario_name] = {
                'total_spend_change': sum(spend_changes.values()) - len(spend_changes),
                'channels_modified': list(spend_changes.keys())
            }
        
        return results
    
    def time_series_decomposition(self):
        """Decompose time series into trend, seasonality"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        ts_data = self.robyn.data.set_index('date')['ticket_sales']
        decomposition = seasonal_decompose(ts_data, model='additive', period=52)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.savefig("robyn_output/time_series_decomposition.png", dpi=300)
        plt.close()
        
        return decomposition
    
    def cross_validation(self, n_splits=5):
        """Time series cross-validation"""
        print("🔄 Running cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        # Prepare data
        media_data = self.robyn.prepare_media_data()
        X = []
        for channel in self.robyn.channels:
            if channel in media_data:
                X.append(media_data[channel]['saturated'])
        
        if X:
            X = np.column_stack(X)
            y = self.robyn.data['ticket_sales'].values
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                score = mean_squared_error(y_test, y_pred, squared=False)
                scores.append(score)
        
        return {'cv_scores': scores, 'mean_rmse': np.mean(scores)}
    
    def channel_interaction_analysis(self):
        """Analyze channel interactions"""
        print("🔗 Analyzing channel interactions...")
        
        # Calculate correlation matrix
        spend_cols = [f"{ch.lower()}_spend" for ch in self.robyn.channels]
        spend_data = self.robyn.data[spend_cols].fillna(0)
        
        correlation_matrix = spend_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Channel Spend Correlations')
        plt.tight_layout()
        plt.savefig("robyn_output/channel_correlations.png", dpi=300)
        plt.close()
        
        return correlation_matrix
    
    def diminishing_returns_analysis(self):
        """Analyze diminishing returns for each channel"""
        print("📉 Analyzing diminishing returns...")
        
        results = {}
        for channel in self.robyn.channels:
            spend_col = f"{channel.lower()}_spend"
            if spend_col in self.robyn.data.columns:
                spend_data = self.robyn.data[spend_col].values
                if np.sum(spend_data) > 0:
                    # Calculate marginal efficiency at different spend levels
                    spend_levels = np.percentile(spend_data[spend_data > 0], [25, 50, 75, 90])
                    marginal_returns = []
                    
                    for level in spend_levels:
                        # Simplified marginal return calculation
                        saturated = self.robyn.saturation_transform([level], alpha=1.5, gamma=0.7)[0]
                        marginal_returns.append(saturated / level if level > 0 else 0)
                    
                    results[channel] = {
                        'spend_levels': spend_levels,
                        'marginal_returns': marginal_returns
                    }
        
        return results

def run_extended_analysis():
    """Run all extended analyses"""
    print("🚀 Running Extended Robyn Analysis")
    
    # Load existing Robyn model
    from robyn_analysis import RobynMMM
    robyn = RobynMMM()
    robyn.load_data("../ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv")
    robyn.fit_model()
    
    # Initialize extensions
    extensions = RobynExtensions(robyn)
    
    # Budget optimization
    optimal_budget = extensions.budget_optimization(1000000)  # €1M budget
    print(f"Optimal allocation: {optimal_budget}")
    
    # Scenario analysis
    scenarios = {
        "increase_google": {"Google": 1.5, "Facebook": 0.8},
        "shift_to_tiktok": {"TikTok": 2.0, "LinkedIn": 0.5}
    }
    scenario_results = extensions.scenario_analysis(scenarios)
    print(f"Scenario results: {scenario_results}")
    
    # Time series analysis
    extensions.time_series_decomposition()
    
    # Cross-validation
    cv_results = extensions.cross_validation()
    print(f"CV RMSE: {cv_results['mean_rmse']:.0f}")
    
    # Channel interactions
    extensions.channel_interaction_analysis()
    
    # Diminishing returns
    diminishing_returns = extensions.diminishing_returns_analysis()
    
    print("✅ Extended analysis complete!")

if __name__ == "__main__":
    run_extended_analysis()