#!/usr/bin/env python3
"""
Check all available Meridian analytics methods
"""

import pickle
import inspect
from meridian.analysis import analyzer, optimizer, visualizer

def check_available_methods():
    print("🔍 AVAILABLE MERIDIAN ANALYTICS METHODS")
    print("=" * 50)
    
    # Check Analyzer methods
    print("\n📊 ANALYZER METHODS:")
    analyzer_methods = [m for m in dir(analyzer.Analyzer) if not m.startswith('_')]
    for method in sorted(analyzer_methods):
        try:
            doc = getattr(analyzer.Analyzer, method).__doc__
            doc_summary = doc.split('\n')[0] if doc else "No documentation"
            print(f"  • {method}: {doc_summary}")
        except:
            print(f"  • {method}")
    
    # Check Optimizer methods
    print("\n🎯 OPTIMIZER METHODS:")
    try:
        optimizer_methods = [m for m in dir(optimizer.BudgetOptimizer) if not m.startswith('_')]
        for method in sorted(optimizer_methods):
            try:
                doc = getattr(optimizer.BudgetOptimizer, method).__doc__
                doc_summary = doc.split('\n')[0] if doc else "No documentation"
                print(f"  • {method}: {doc_summary}")
            except:
                print(f"  • {method}")
    except:
        print("  ⚠ BudgetOptimizer not available")
    
    # Check Visualizer methods
    print("\n📈 VISUALIZER METHODS:")
    visualizer_classes = [c for c in dir(visualizer) if c[0].isupper() and not c.startswith('_')]
    for cls_name in visualizer_classes:
        try:
            cls = getattr(visualizer, cls_name)
            print(f"  • {cls_name} class:")
            methods = [m for m in dir(cls) if not m.startswith('_')]
            for method in sorted(methods)[:5]:  # Show first 5 methods
                try:
                    doc = getattr(cls, method).__doc__
                    doc_summary = doc.split('\n')[0] if doc else "No documentation"
                    print(f"    - {method}: {doc_summary}")
                except:
                    print(f"    - {method}")
            if len(methods) > 5:
                print(f"    - ... and {len(methods) - 5} more methods")
        except:
            print(f"  • {cls_name} (could not inspect)")
    
    # Check methods currently used in your script
    print("\n🔄 METHODS CURRENTLY USED IN YOUR SCRIPT:")
    used_methods = [
        "incremental_outcome",
        "roi",
        "baseline_summary_metrics"
    ]
    for method in used_methods:
        print(f"  • {method}")
    
    # Identify unused but potentially useful methods
    print("\n⭐ UNUSED BUT POTENTIALLY USEFUL METHODS:")
    potentially_useful = [
        "adstock_decay: Computes the adstock decay for each media channel.",
        "cpik: Computes the cost per incremental KPI for each media channel.",
        "expected_outcome: Computes the expected outcome for each geo and time period.",
        "expected_vs_actual_data: Computes the expected vs actual data.",
        "hill_curves: Computes the hill curves for each media channel.",
        "marginal_roi: Computes the marginal ROI for each media channel.",
        "optimal_freq: Computes the optimal frequency for each media channel.",
        "predictive_accuracy: Computes the predictive accuracy of the model.",
        "response_curves: Computes the response curves for each media channel."
    ]
    for method in potentially_useful:
        print(f"  • {method}")
    
    # Check if a fitted model is available to test methods
    print("\n🧪 TESTING WITH FITTED MODEL:")
    try:
        with open('/Users/aaronmeagher/Work/google_meridian/google/test/ticket_analysis/ticket_sales_model.pkl', 'rb') as f:
            mmm = pickle.load(f)
        print("  ✓ Model loaded successfully")
        
        # Create analyzer
        analyzer_obj = analyzer.Analyzer(mmm)
        print("  ✓ Analyzer created successfully")
        
        # Test a few methods
        print("\n  Testing methods:")
        
        # Test marginal_roi
        try:
            result = analyzer_obj.marginal_roi(use_kpi=True)
            print(f"  ✓ marginal_roi: {type(result)} with shape {result.shape if hasattr(result, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"  ✗ marginal_roi failed: {e}")
        
        # Test adstock_decay
        try:
            result = analyzer_obj.adstock_decay()
            print(f"  ✓ adstock_decay: {type(result)} with shape {result.shape if hasattr(result, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"  ✗ adstock_decay failed: {e}")
        
        # Test hill_curves
        try:
            result = analyzer_obj.hill_curves()
            print(f"  ✓ hill_curves: {type(result)} with shape {result.shape if hasattr(result, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"  ✗ hill_curves failed: {e}")
        
        # Test cpik
        try:
            result = analyzer_obj.cpik(use_kpi=True)
            print(f"  ✓ cpik: {type(result)} with shape {result.shape if hasattr(result, 'shape') else 'unknown'}")
        except Exception as e:
            print(f"  ✗ cpik failed: {e}")
        
    except Exception as e:
        print(f"  ✗ Could not load model: {e}")

if __name__ == "__main__":
    check_available_methods()