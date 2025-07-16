#!/usr/bin/env python3
"""
Meridian Model Insights
======================
Extract insights from fitted model that aren't visible in raw data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from meridian.analysis import analyzer

def load_fitted_model():
    """Load a fitted model"""
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/better_priors/Conservative_model.pkl', 'rb') as f:
            mmm = pickle.load(f)
        return mmm
    except:
        print("No fitted model found. Run meridian_success_grid_2.py first.")
        return None

def extract_model_insights(mmm):
    """Extract insights only available from fitted model"""
    print("🔍 MERIDIAN MODEL INSIGHTS")
    print("=" * 50)
    
    analyzer_obj = analyzer.Analyzer(mmm)
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # 1. INCREMENTAL CONTRIBUTION (not visible in raw data)
    print("1. 📈 INCREMENTAL MEDIA CONTRIBUTION:")
    print("   (How much each channel actually drove clicks)")
    try:
        # Try different method names for media contribution
        if hasattr(analyzer_obj, 'get_posterior_metrics'):
            metrics = analyzer_obj.get_posterior_metrics()
            print(f"   Available metrics: {list(metrics.keys()) if hasattr(metrics, 'keys') else 'Unknown format'}")
        elif hasattr(mmm, 'posterior_samples'):
            print("   Model has posterior samples - contribution analysis possible")
            print("   ⚠ Requires manual extraction from posterior samples")
        else:
            print("   ⚠ No standard contribution methods available")
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")
    
    # 2. BASELINE vs INCREMENTAL SPLIT
    print("\n2. 🏗️ BASELINE vs INCREMENTAL SPLIT:")
    print("   (What would happen without media)")
    try:
        # Check available analyzer methods
        analyzer_methods = [method for method in dir(analyzer_obj) if not method.startswith('_')]
        print(f"   Available analyzer methods: {analyzer_methods[:5]}...")  # Show first 5
        
        # Try to access model components directly
        if hasattr(mmm, 'input_data'):
            total_kpi = np.sum(mmm.input_data.kpi)
            print(f"   Total observed KPI: {total_kpi:,.0f} clicks")
            print(f"   ⚠ Baseline extraction requires specific model analysis")
        
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")
    
    # 3. SATURATION CURVES (diminishing returns)
    print("\n3. 📉 SATURATION ANALYSIS:")
    print("   (Diminishing returns by channel)")
    try:
        # This would show how close each channel is to saturation
        print("   Channel saturation levels:")
        print("   (Higher = more saturated, lower efficiency at current spend)")
        # Note: Actual saturation extraction depends on model version
        print("   ⚠ Saturation curves require additional analysis")
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")
    
    # 4. ADSTOCK/CARRYOVER EFFECTS
    print("\n4. 🔄 CARRYOVER EFFECTS:")
    print("   (How long media effects last)")
    try:
        # Adstock shows how media effects decay over time
        print("   Media carryover analysis:")
        print("   (Shows persistence of media effects beyond immediate period)")
        print("   ⚠ Carryover analysis requires additional model parameters")
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")
    
    # 5. COUNTERFACTUAL SCENARIOS
    print("\n5. 🎯 WHAT-IF SCENARIOS:")
    print("   (Model can predict outcomes of different spend levels)")
    try:
        print("   Possible analyses:")
        print("   • What if we doubled Facebook spend?")
        print("   • What if we reallocated LinkedIn budget to Google?")
        print("   • What's the optimal budget allocation?")
        print("   ⚠ Scenario analysis requires optimization module")
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")
    
    # 6. STATISTICAL SIGNIFICANCE
    print("\n6. 📊 STATISTICAL CONFIDENCE:")
    print("   (Uncertainty around estimates)")
    try:
        print("   Model provides confidence intervals for:")
        print("   • Each channel's true contribution")
        print("   • ROI estimates with uncertainty bands")
        print("   • Probability that one channel outperforms another")
        print("   ⚠ Confidence intervals require posterior sample analysis")
    except Exception as e:
        print(f"   ⚠ Could not extract: {e}")

def create_insight_plots(mmm):
    """Create plots showing model-specific insights"""
    print("\n📊 CREATING INSIGHT PLOTS...")
    
    analyzer_obj = analyzer.Analyzer(mmm)
    plot_dir = '/Users/aaronmeagher/Work/google_meridian/google/test/better_priors'
    channels = ['Google', 'Facebook', 'LinkedIn', 'Reddit', 'Bing', 'TikTok', 'Twitter', 'Instagram']
    
    # Plot 1: Model vs Observed KPI
    try:
        # Create a simple comparison plot using available data
        plt.figure(figsize=(12, 6))
        
        # Plot observed KPI over time
        total_kpi_time = np.mean(mmm.input_data.kpi, axis=0)
        plt.plot(total_kpi_time, label='Observed KPI', linewidth=2)
        
        # Add trend line to show general pattern
        x = np.arange(len(total_kpi_time))
        z = np.polyfit(x, total_kpi_time, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), label='Trend', linestyle='--', alpha=0.7)
        
        plt.title('KPI Over Time (Model Input)')
        plt.xlabel('Time Period')
        plt.ylabel('Clicks')
        plt.legend()
        
        path = f'{plot_dir}/Model_Insight_KPI_Analysis.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: Model_Insight_KPI_Analysis.png")
        
    except Exception as e:
        print(f"  ⚠ KPI plot failed: {e}")
    
    # Plot 2: Model Information Summary
    try:
        plt.figure(figsize=(12, 8))
        
        # Create an information plot about the model
        plt.text(0.1, 0.9, 'MERIDIAN MODEL ANALYSIS', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
        
        # Model info
        model_info = [
            f'Model Type: Meridian MMM',
            f'Channels: {len(channels)}',
            f'Time Periods: {mmm.input_data.kpi.shape[1]}',
            f'Total KPI: {np.sum(mmm.input_data.kpi):,.0f} clicks',
            f'Total Spend: €{np.sum(mmm.input_data.media_spend):,.0f}',
            '',
            'Model Capabilities:',
            '• Causal attribution of media effects',
            '• Baseline vs incremental analysis', 
            '• Saturation curve modeling',
            '• Carryover effect estimation',
            '• Budget optimization scenarios',
            '',
            'Note: Advanced insights require specific',
            'analysis methods beyond basic model fitting.'
        ]
        
        for i, info in enumerate(model_info):
            plt.text(0.1, 0.85 - i*0.05, info, fontsize=10, transform=plt.gca().transAxes)
        
        plt.axis('off')
        plt.title('Meridian Model Summary')
        
        path = f'{plot_dir}/Model_Insight_Summary.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: Model_Insight_Summary.png")
        
    except Exception as e:
        print(f"  ⚠ Summary plot failed: {e}")

def main():
    mmm = load_fitted_model()
    if mmm is None:
        return
    
    extract_model_insights(mmm)
    create_insight_plots(mmm)
    
    print("\n🎯 KEY INSIGHTS FROM MERIDIAN MODEL:")
    print("Unlike raw data analysis, the model provides:")
    print("• Causal attribution (not just correlation)")
    print("• Incremental lift from each channel")
    print("• Baseline performance without media")
    print("• Statistical confidence in estimates")
    print("• Ability to predict counterfactual scenarios")
    print("• Account for adstock/carryover effects")
    print("• Diminishing returns (saturation) analysis")

if __name__ == "__main__":
    main()