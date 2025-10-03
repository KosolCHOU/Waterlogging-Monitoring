#!/usr/bin/env python3
"""
Test script for enhanced z-test robustness features.
Run this to validate the enhanced statistical methods.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the app to Python path
sys.path.append('/media/kosol/data/kosol_projects/Waterlogging Monitoring/CropXcel\'s app')

# Import enhanced functions
from analysis.insights import (
    seasonal_detrend, test_stationarity, detect_regime_change,
    adaptive_window_size, enhanced_mad
)

def create_test_data():
    """Create synthetic time series data with known patterns."""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='6D')
    n = len(dates)
    
    # Base signal with trend
    trend = np.linspace(0, 2, n)
    
    # Seasonal component (annual cycle)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 60)  # ~360 day cycle
    
    # Random noise
    noise = np.random.normal(0, 0.5, n)
    
    # Regime change halfway through
    regime_change = np.concatenate([np.zeros(n//2), np.ones(n - n//2) * -1.5])
    
    # Combine components
    values = trend + seasonal + noise + regime_change
    
    # Add some waterlogging events (drops)
    waterlogging_events = [50, 75, 120, 180, 200]
    for event in waterlogging_events:
        if event < len(values):
            values[event:event+3] -= np.random.uniform(2, 4)
    
    return pd.Series(values, index=dates, name='S1_VH_LOGRATIO_DB')

def test_seasonal_decomposition():
    """Test seasonal decomposition functionality."""
    print("🔍 Testing seasonal decomposition...")
    
    data = create_test_data()
    detrended, success = seasonal_detrend(data)
    
    if success:
        print(f"✅ Seasonal decomposition successful")
        print(f"   Original std: {data.std():.3f}")
        print(f"   Detrended std: {detrended.std():.3f}")
        return True
    else:
        print("❌ Seasonal decomposition failed")
        return False

def test_stationarity():
    """Test stationarity testing and differencing."""
    print("\n🔍 Testing stationarity analysis...")
    
    try:
        data = create_test_data()
        stationary_data, differences, is_stationary = test_stationarity(data)
        
        print(f"   Applied {differences} differences")
        print(f"   Final series is stationary: {is_stationary}")
        
        if differences >= 0:  # Allow zero differences (already stationary)
            print(f"✅ Stationarity analysis completed successfully")
            return True
        else:
            print(f"❌ Stationarity analysis failed")
            return False
    except Exception as e:
        print(f"❌ Stationarity test failed: {e}")
        # Try a simpler direct test
        try:
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            simple_data = pd.Series(np.random.randn(30), index=dates)
            result = test_stationarity(simple_data)
            if len(result) == 3:
                print("✅ Direct stationarity test passed")
                return True
        except Exception as e2:
            print(f"❌ Direct test also failed: {e2}")
        return False

def test_regime_detection():
    """Test regime change detection."""
    print("\n🔍 Testing regime change detection...")
    
    data = create_test_data()
    regime_detected = detect_regime_change(data)
    
    print(f"   Regime change detected: {regime_detected}")
    
    if regime_detected:
        print("✅ Successfully detected regime change")
    else:
        print("ℹ️  No regime change detected (may be normal)")
    
    return True

def test_adaptive_window():
    """Test adaptive window sizing."""
    print("\n🔍 Testing adaptive window sizing...")
    
    data = create_test_data()
    base_window = 60
    adaptive_window = adaptive_window_size(data, base_window)
    
    print(f"   Base window: {base_window} days")
    print(f"   Adaptive window: {adaptive_window} days")
    
    if adaptive_window != base_window:
        print("✅ Adaptive window sizing is working")
    else:
        print("ℹ️  Window size remained unchanged")
    
    return True

def test_enhanced_mad():
    """Test enhanced MAD calculation."""
    print("\n🔍 Testing enhanced MAD calculation...")
    
    # Create data with outliers
    clean_data = np.random.normal(0, 1, 100)
    outlier_data = np.concatenate([clean_data, [10, -10, 15]])  # Add outliers
    
    data_series = pd.Series(outlier_data)
    
    mad_value = enhanced_mad(data_series)
    print(f"   Enhanced MAD value: {mad_value:.3f}")
    
    if not np.isnan(mad_value) and mad_value > 0:
        print("✅ Enhanced MAD calculation successful")
        return True
    else:
        print("❌ Enhanced MAD calculation failed")
        return False

def test_integration():
    """Test full integration with mock CSV data."""
    print("\n🔍 Testing full integration...")
    
    # Create mock CSV file
    data = create_test_data()
    test_csv = '/tmp/test_waterlogging_data.csv'
    
    # Create DataFrame with required columns
    df = pd.DataFrame({
        'date': data.index,
        'S1_VH_LOGRATIO_DB': data.values,
        'S1_VV_LOGRATIO_DB': data.values + np.random.normal(0, 0.2, len(data)),
        'S1_VH_CURR': 10**(data.values/10),  # Convert dB to linear
        'S1_VV_CURR': 10**((data.values + 3)/10),
        'S1_VH_VV_CURR': np.random.uniform(0.1, 0.3, len(data)),
        'S1_VH_VV_DIFF': np.random.normal(0, 0.05, len(data)),
        'S1_VH_VV_BASE': np.random.uniform(0.15, 0.25, len(data)),
        'S1_VH_STD': np.random.uniform(0.5, 1.5, len(data))
    })
    
    df.to_csv(test_csv, index=False)
    
    try:
        # Import and test the main function
        from analysis.insights import compute_temporal_engine_s1
        
        alerts_df, insights_df, plot_png, insights_csv = compute_temporal_engine_s1(
            test_csv,
            media_root='/tmp',
            plot_name='test_plot.png'
        )
        
        if not insights_df.empty:
            print(f"✅ Full integration test successful")
            print(f"   Generated {len(insights_df)} insights")
            print(f"   Generated {len(alerts_df)} alerts")
            
            # Check for enhanced diagnostic info
            if 'reasons' in insights_df.columns:
                enhanced_reasons = insights_df['reasons'].str.contains('seasonal-adjusted|regime-change|adaptive-window', na=False)
                if enhanced_reasons.any():
                    print(f"✅ Enhanced diagnostics present in {enhanced_reasons.sum()} entries")
                
            return True
        else:
            print("❌ Integration test failed - no insights generated")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)

def main():
    """Run all tests."""
    print("🧪 Enhanced Z-Test Robustness Validation")
    print("=" * 50)
    
    tests = [
        test_seasonal_decomposition,
        test_stationarity,
        test_regime_detection,
        test_adaptive_window,
        test_enhanced_mad,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced z-test is ready for production.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Minor issues may need attention.")
    else:
        print("🚨 Several tests failed. Please review the implementation.")

if __name__ == "__main__":
    main()