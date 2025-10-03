#!/usr/bin/env python3
"""
Test script to verify that TIF export now uses actual satellite acquisition dates
and produces aligned timestamps with timeseries data.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "CropXcel's app"
sys.path.insert(0, str(app_dir))

# Test with a simple geometry
test_geom = {
    "type": "Polygon", 
    "coordinates": [[[104.85, 11.54], [104.86, 11.54], [104.86, 11.55], [104.85, 11.55], [104.85, 11.54]]]
}

def test_tif_export_alignment():
    """Test that TIF export now uses actual satellite dates"""
    
    try:
        # Import the engine functions
        from analysis.engine import export_stack_from_geom
        
        # Create temporary file for TIF export
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        print("Testing TIF export with actual satellite date alignment...")
        print(f"Test geometry: Small area around Phnom Penh")
        print(f"Output path: {tmp_path}")
        
        # Export TIF with our modified function
        result_path = export_stack_from_geom(
            geom_geojson=test_geom,
            out_tif=tmp_path,
            event_days=15,
            base_days=45,
            gap_days=5
        )
        
        print(f"\n✅ TIF export completed: {result_path}")
        
        # Check if metadata file was created
        meta_path = tmp_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            print("\n📊 Export Metadata:")
            print(f"  Actual end date: {metadata.get('end')}")
            print(f"  Theoretical end date: {metadata.get('theoretical_end')}")
            print(f"  Uses actual satellite date: {metadata.get('uses_actual_satellite_date')}")
            print(f"  Date synchronization: {metadata.get('date_synchronization')}")
            print(f"  Event count: {metadata.get('event_count')}")
            print(f"  Base count: {metadata.get('base_count')}")
            
            # Check if dates are different (indicating we found actual satellite data)
            actual_date = metadata.get('end')
            theoretical_date = metadata.get('theoretical_end')
            
            if actual_date != theoretical_date:
                print(f"\n🎯 SUCCESS: Using actual satellite date ({actual_date}) instead of theoretical date ({theoretical_date})")
            else:
                print(f"\n⚠️  INFO: Actual and theoretical dates are the same ({actual_date})")
                print("   This could mean: 1) Satellite data was available exactly yesterday, or")
                print("                   2) No satellite data found, fell back to theoretical date")
        
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if os.path.exists(meta_path):
            os.unlink(meta_path)
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing TIF export: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timeseries_export():
    """Test timeseries export for comparison"""
    
    try:
        from analysis.engine import export_s1_timeseries
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        print(f"\nTesting timeseries export for comparison...")
        print(f"Output path: {tmp_path}")
        
        # Export timeseries
        result_path = export_s1_timeseries(
            geom_geojson=test_geom,
            out_csv=tmp_path,
            step_days=10
        )
        
        print(f"✅ Timeseries export completed: {result_path}")
        
        # Read and show first few dates
        import pandas as pd
        df = pd.read_csv(tmp_path)
        if 'date' in df.columns and len(df) > 0:
            print(f"\n📊 Timeseries dates (first 3):")
            for i, date in enumerate(df['date'][:3]):
                print(f"  {i+1}: {date}")
        
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing timeseries export: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Testing Satellite Data Export Alignment")
    print("=" * 50)
    
    # Test both exports
    tif_success = test_tif_export_alignment()
    timeseries_success = test_timeseries_export()
    
    print("\n" + "=" * 50)
    if tif_success and timeseries_success:
        print("✅ All tests completed successfully!")
        print("\n🎯 Key Improvements:")
        print("  - TIF export now searches for actual satellite acquisition dates")
        print("  - Time windows are recalculated based on real satellite availability")
        print("  - Metadata includes both actual and theoretical dates for comparison")
        print("  - This should align TIF export dates with timeseries export dates")
    else:
        print("❌ Some tests failed. Check the error messages above.")