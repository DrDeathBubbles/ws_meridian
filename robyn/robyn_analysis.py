#!/usr/bin/env python3
"""
Meta Robyn Marketing Mix Modeling Implementation
Alternative to BMMM analysis using Facebook's open-source Robyn approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Robyn-style MMM implementation
class RobynMMM:
    def __init__(self):
        self.data = None
        self.model_results = {}
        self.channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
        
    def load_data(self, file_path):
        """Load and prepare data for Robyn analysis"""
        print("📊 Loading data for Robyn MMM...")
        
        # Load the CSV data
        df = pd.read_csv(file_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date_id'])
        
        # Use ticket_sales as KPI (dependent variable)
        daily_data = df.groupby('date').agg({
            'ticket_sales': 'sum',  # Use as KPI
            'campaign_spend_eur': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'reach': 'sum',
            'reddit_spend': 'sum',
            'linkedin_spend': 'sum', 
            'facebook_spend': 'sum',
            'google_spend': 'sum',
            'bing_spend': 'sum',
            'tiktok_spend': 'sum',
            'twitter_spend': 'sum',
            'instagram_spend': 'sum',
            'reddit_impressions': 'sum',
            'linkedin_impressions': 'sum',
            'facebook_impressions': 'sum', 
            'google_impressions': 'sum',
            'bing_impressions': 'sum',
            'tiktok_impressions': 'sum',
            'twitter_impressions': 'sum',
            'instagram_impressions': 'sum'
        }).reset_index()
        
        self.data = daily_data
        print(f"✅ Data loaded: {len(daily_data)} time periods")
        print(f"📊 Total spend: €{daily_data['campaign_spend_eur'].sum():,.0f}")
        print(f"🎫 Total ticket sales: {daily_data['ticket_sales'].sum():,}")
        return daily_data
    
    def adstock_transform(self, x, adstock_rate=0.5):
        """Apply adstock transformation (carryover effect)"""
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]
        
        for i in range(1, len(x)):
            adstocked[i] = x[i] + adstock_rate * adstocked[i-1]
        
        return adstocked
    
    def saturation_transform(self, x, alpha=2.0, gamma=0.5):
        """Apply saturation transformation (diminishing returns)"""
        return alpha * (x ** gamma) / (alpha + x ** gamma)
    
    def hill_transform(self, x, ec=0.5, slope=2.0):
        """Hill transformation for saturation curves"""
        return 1 / (1 + (ec / x) ** slope)
    
    def prepare_media_data(self):
        """Prepare media data with transformations"""
        print("🔧 Applying media transformations...")
        
        media_data = {}
        
        for channel in self.channels:
            spend_col = f"{channel.lower()}_spend"
            if spend_col in self.data.columns:
                spend = self.data[spend_col].values
                
                # Only process channels with actual spend
                if np.sum(spend) > 0:
                    # Apply adstock transformation
                    adstocked = self.adstock_transform(spend, adstock_rate=0.3)
                    
                    # Apply saturation transformation
                    saturated = self.saturation_transform(adstocked, alpha=1.5, gamma=0.7)
                    
                    media_data[channel] = {
                        'raw_spend': spend,
                        'adstocked': adstocked,
                        'saturated': saturated
                    }
        
        print(f"📊 Processed {len(media_data)} channels with spend data")
        return media_data
    
    def fit_model(self):
        """Fit the Robyn MMM model"""
        print("🎯 Fitting Robyn MMM model...")
        
        # Prepare media data
        media_data = self.prepare_media_data()
        
        # Simple linear regression approach (Robyn uses more sophisticated methods)
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Prepare feature matrix
        X = []
        feature_names = []
        
        for channel in self.channels:
            if channel in media_data:
                X.append(media_data[channel]['saturated'])
                feature_names.append(f"{channel}_saturated")
        
        if not X:
            print("❌ No media data found!")
            return
        
        X = np.column_stack(X)
        y = self.data['ticket_sales'].values
        
        # Fit model
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Store results
        self.model_results = {
            'model': model,
            'coefficients': dict(zip(feature_names, model.coef_)),
            'intercept': model.intercept_,
            'r2_score': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'predictions': y_pred,
            'actual': y,
            'media_data': media_data,
            'feature_names': feature_names
        }
        
        print(f"✅ Model fitted - R²: {self.model_results['r2_score']:.3f}")
        
    def calculate_contribution(self):
        """Calculate channel contributions"""
        if not self.model_results:
            print("❌ Model not fitted yet!")
            return
        
        print("📈 Calculating channel contributions...")
        
        contributions = {}
        media_data = self.model_results['media_data']
        coefficients = self.model_results['coefficients']
        
        for channel in self.channels:
            if channel in media_data:
                feature_name = f"{channel}_saturated"
                if feature_name in coefficients:
                    saturated_values = media_data[channel]['saturated']
                    contribution = saturated_values * coefficients[feature_name]
                    contributions[channel] = {
                        'total_contribution': np.sum(contribution),
                        'contribution_series': contribution,
                        'coefficient': coefficients[feature_name]
                    }
        
        self.model_results['contributions'] = contributions
        return contributions
    
    def calculate_roi(self):
        """Calculate ROI by channel"""
        if 'contributions' not in self.model_results:
            self.calculate_contribution()
        
        print("💰 Calculating ROI by channel...")
        
        roi_results = {}
        contributions = self.model_results['contributions']
        
        for channel in self.channels:
            spend_col = f"{channel.lower()}_spend"
            if spend_col in self.data.columns and channel in contributions:
                total_spend = self.data[spend_col].sum()
                total_contribution = contributions[channel]['total_contribution']
                
                if total_spend > 0:
                    roi = total_contribution / total_spend
                    roi_results[channel] = {
                        'roi': roi,
                        'total_spend': total_spend,
                        'total_contribution': total_contribution
                    }
        
        self.model_results['roi'] = roi_results
        return roi_results
    
    def plot_model_fit(self, output_dir="robyn_output"):
        """Plot model fit diagnostics"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.model_results:
            print("❌ Model not fitted yet!")
            return
        
        actual = self.model_results['actual']
        predicted = self.model_results['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0,0].scatter(actual, predicted, alpha=0.6)
        axes[0,0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Ticket Sales')
        axes[0,0].set_ylabel('Predicted Ticket Sales')
        axes[0,0].set_title(f'Actual vs Predicted (R² = {self.model_results["r2_score"]:.3f})')
        
        # Time series
        dates = self.data['date']
        axes[0,1].plot(dates, actual, label='Actual', linewidth=2)
        axes[0,1].plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8)
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Ticket Sales')
        axes[0,1].set_title('Time Series: Actual vs Predicted')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Residuals
        residuals = actual - predicted
        axes[1,0].scatter(predicted, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot')
        
        # Residuals histogram
        axes[1,1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Residuals')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Residuals Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_fit_diagnostics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: model_fit_diagnostics.png")
    
    def plot_channel_contribution(self, output_dir="robyn_output"):
        """Plot channel contributions"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if 'contributions' not in self.model_results:
            self.calculate_contribution()
        
        contributions = self.model_results['contributions']
        
        # Total contribution by channel
        channels = list(contributions.keys())
        total_contribs = [contributions[ch]['total_contribution'] for ch in channels]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(channels, total_contribs, color=plt.cm.Set3(np.linspace(0, 1, len(channels))))
        plt.title('Total Channel Contribution (Robyn MMM)', fontsize=16)
        plt.xlabel('Channel')
        plt.ylabel('Total Contribution (Ticket Sales)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, total_contribs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_contribs)*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/channel_contribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: channel_contribution.png")
    
    def plot_roi_analysis(self, output_dir="robyn_output"):
        """Plot ROI analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if 'roi' not in self.model_results:
            self.calculate_roi()
        
        roi_data = self.model_results['roi']
        
        channels = list(roi_data.keys())
        roi_values = [roi_data[ch]['roi'] for ch in channels]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(channels, roi_values, color=plt.cm.viridis(np.linspace(0, 1, len(channels))))
        plt.title('ROI by Channel (Robyn MMM)', fontsize=16)
        plt.xlabel('Channel')
        plt.ylabel('ROI (Tickets per €)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, roi_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(roi_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roi_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: roi_analysis.png")
    
    def plot_response_curves(self, output_dir="robyn_output"):
        """Plot response curves for each channel"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if 'contributions' not in self.model_results:
            self.calculate_contribution()
        
        media_data = self.model_results['media_data']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, channel in enumerate(self.channels[:8]):
            ax = axes[i]
            
            if channel in media_data:
                spend_col = f"{channel.lower()}_spend"
                if spend_col in self.data.columns:
                    raw_spend = media_data[channel]['raw_spend']
                    saturated = media_data[channel]['saturated']
                    
                    # Create response curve
                    max_spend = np.max(raw_spend) * 2
                    spend_range = np.linspace(0, max_spend, 100)
                    
                    # Apply same transformations
                    adstocked_range = self.adstock_transform(spend_range, adstock_rate=0.3)
                    saturated_range = self.saturation_transform(adstocked_range, alpha=1.5, gamma=0.7)
                    
                    ax.plot(spend_range, saturated_range, 'b-', linewidth=2)
                    
                    # Mark current spend level
                    current_spend = np.mean(raw_spend[raw_spend > 0]) if np.any(raw_spend > 0) else 0
                    if current_spend > 0:
                        current_response = self.saturation_transform(
                            self.adstock_transform([current_spend], adstock_rate=0.3)[0], 
                            alpha=1.5, gamma=0.7
                        )
                        ax.scatter([current_spend], [current_response], color='red', s=100, zorder=5)
                    
                    ax.set_title(f'{channel} Response Curve')
                    ax.set_xlabel('Spend (€)')
                    ax.set_ylabel('Saturated Response')
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{channel} Response Curve')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: response_curves.png")
    
    def generate_summary_report(self, output_dir="robyn_output"):
        """Generate summary report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.model_results:
            print("❌ Model not fitted yet!")
            return
        
        # Calculate all metrics
        self.calculate_contribution()
        self.calculate_roi()
        
        report = []
        report.append("=" * 60)
        report.append("ROBYN MARKETING MIX MODEL - SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {self.data['date'].min()} to {self.data['date'].max()}")
        report.append(f"Total Observations: {len(self.data)}")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 20)
        report.append(f"R-squared: {self.model_results['r2_score']:.3f}")
        report.append(f"Mean Absolute Error: {self.model_results['mae']:.2f}")
        report.append(f"Baseline (Intercept): {self.model_results['intercept']:.2f}")
        report.append("")
        
        # Channel Performance
        report.append("CHANNEL PERFORMANCE:")
        report.append("-" * 20)
        
        if 'roi' in self.model_results:
            roi_data = self.model_results['roi']
            sorted_channels = sorted(roi_data.items(), key=lambda x: x[1]['roi'], reverse=True)
            
            for channel, metrics in sorted_channels:
                report.append(f"{channel}:")
                report.append(f"  ROI: {metrics['roi']:.3f} tickets/€")
                report.append(f"  Total Spend: €{metrics['total_spend']:,.0f}")
                report.append(f"  Total Contribution: {metrics['total_contribution']:.0f} tickets")
                report.append("")
        
        # Coefficients
        report.append("MODEL COEFFICIENTS:")
        report.append("-" * 20)
        coefficients = self.model_results['coefficients']
        for feature, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
            report.append(f"{feature}: {coef:.4f}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(f"{output_dir}/robyn_summary_report.txt", 'w') as f:
            f.write(report_text)
        
        print("📋 ROBYN MMM SUMMARY:")
        print(report_text)
        print(f"\n✅ Full report saved: robyn_summary_report.txt")


def run_robyn_analysis():
    """Main function to run Robyn MMM analysis"""
    print("🚀 Starting Meta Robyn MMM Analysis")
    print("=" * 50)
    
    # Initialize Robyn MMM
    robyn = RobynMMM()
    
    # Load data
    # data_path = "../ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv"
    # Updated to absolute path for reliability
    data_path = "/home/aaron/work/ws_meridian/ws_attempt/data/raw_data/regularized_web_summit_data_fixed.csv"
    try:
        robyn.load_data(data_path)
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists in the correct location.")
        return
    
    # Fit model
    robyn.fit_model()
    
    # Generate all plots and analysis
    output_dir = "robyn_output"
    robyn.plot_model_fit(output_dir)
    robyn.plot_channel_contribution(output_dir)
    robyn.plot_roi_analysis(output_dir)
    robyn.plot_response_curves(output_dir)
    
    # Generate summary report
    robyn.generate_summary_report(output_dir)
    
    print("\n🎉 Robyn MMM Analysis Complete!")
    print(f"All results saved in: {output_dir}/")


if __name__ == "__main__":
    run_robyn_analysis()