#!/usr/bin/env python3
"""
Meridian Version Check
=====================
Check current version and requirements for full analysis
"""

def check_versions():
    print("🔍 MERIDIAN VERSION CHECK")
    print("=" * 50)
    
    # Check current Meridian version
    try:
        import meridian
        if hasattr(meridian, '__version__'):
            current_version = meridian.__version__
            print(f"Current Meridian version: {current_version}")
        else:
            print("Current Meridian version: Unknown (no __version__ attribute)")
            
        # Check installation method
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution('meridian')
            print(f"Installed via: {dist.location}")
            print(f"Package info: {dist}")
        except:
            print("Package info: Not available")
            
    except ImportError:
        print("❌ Meridian not installed")
        return
    
    # Check available analysis modules
    print(f"\n📦 AVAILABLE MODULES:")
    modules_to_check = [
        'meridian.analysis.analyzer',
        'meridian.analysis.visualizer', 
        'meridian.analysis.optimizer',
        'meridian.analysis.summarizer'
    ]
    
    for module_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[''])
            classes = [name for name in dir(module) if not name.startswith('_') and name[0].isupper()]
            print(f"✓ {module_name}: {classes[:3]}...")
        except ImportError as e:
            print(f"❌ {module_name}: Not available")
    
    # Check analyzer methods specifically
    print(f"\n🔬 ANALYZER METHODS:")
    try:
        from meridian.analysis import analyzer
        # Create a dummy analyzer to check methods
        methods = [method for method in dir(analyzer.Analyzer) if not method.startswith('_')]
        print(f"Available methods: {len(methods)}")
        
        # Check for key methods
        key_methods = [
            'get_media_contribution',
            'get_baseline', 
            'get_posterior_metrics',
            'get_summary',
            'get_results'
        ]
        
        for method in key_methods:
            if method in methods:
                print(f"✓ {method}: Available")
            else:
                print(f"❌ {method}: Missing")
                
    except Exception as e:
        print(f"❌ Analyzer check failed: {e}")
    
    # Version recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("For full MMM analysis capabilities, you need:")
    print("• Meridian version 0.1.0+ (latest from GitHub)")
    print("• Full analysis module with get_media_contribution()")
    print("• Posterior sample access for confidence intervals")
    print("• Optimization module for budget allocation")
    
    print(f"\n🔧 UPGRADE OPTIONS:")
    print("1. Install latest from GitHub:")
    print("   pip install git+https://github.com/google/meridian.git")
    print("2. Check for newer PyPI releases:")
    print("   pip install --upgrade meridian")
    print("3. Install development version:")
    print("   pip install meridian[dev]")

if __name__ == "__main__":
    check_versions()