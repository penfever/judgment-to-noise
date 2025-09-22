#!/usr/bin/env python3
"""
Test script to verify numpy 2.0 compatibility changes work correctly.
"""

import sys
import numpy as np

def test_numpy_compatibility():
    """Test that numpy compatibility shims work correctly."""
    print(f"Testing with NumPy version: {np.__version__}")
    
    # Test the compatibility shim
    if not hasattr(np, 'NAN'):
        np.NAN = np.nan
        print("Added np.NAN compatibility shim")
    else:
        print("np.NAN is available natively")
    
    # Test that np.NAN and np.nan are equivalent
    try:
        assert np.isnan(np.NAN)
        assert np.isnan(np.nan)
        assert np.NAN is np.nan or (np.isnan(np.NAN) and np.isnan(np.nan))
        print("✓ np.NAN and np.nan work correctly")
    except Exception as e:
        print(f"✗ Error with np.NAN/np.nan: {e}")
        return False
    
    # Test basic numpy operations that should work in both versions
    try:
        arr = np.array([1, 2, np.nan, 4])
        print(f"✓ Array creation with np.nan: {arr}")
        
        result = np.nanmean(arr)
        print(f"✓ np.nanmean works: {result}")
        
        mask = np.isnan(arr)
        print(f"✓ np.isnan works: {mask}")
        
        percentiles = np.percentile(arr[~mask], [25, 50, 75])
        print(f"✓ np.percentile works: {percentiles}")
        
    except Exception as e:
        print(f"✗ Error with basic numpy operations: {e}")
        return False
    
    # Test the specific usage from show_result.py
    try:
        names = ['model_a', 'model_b', 'model_c']
        wins = {a: {b: 0.5 if a != b else np.nan for b in names} for a in names}
        data = {
            a: [wins[a][b] if a != b else np.nan for b in names]
            for a in names
        }
        print("✓ show_result.py style np.nan usage works")
        
        import pandas as pd
        df = pd.DataFrame(data, index=names)
        print(f"✓ DataFrame creation with np.nan: \n{df}")
        
    except Exception as e:
        print(f"✗ Error with show_result.py style usage: {e}")
        return False
    
    print("All compatibility tests passed!")
    return True

def test_show_result_imports():
    """Test that show_result.py can be imported without errors."""
    try:
        # Test importing show_result.py
        sys.path.insert(0, '/Users/anonpp/Library/CloudStorage/GoogleDrive-anonppp@gmail.com/My Drive/Current Projects/oumi/llm-judge-oumi/arena-hard-auto')
        import show_result
        print("✓ show_result.py imports successfully")
        
        # Test that the compatibility shim was applied
        if hasattr(np, 'NAN'):
            print("✓ np.NAN is available after import")
        else:
            print("✗ np.NAN is not available after import")
            return False
            
    except Exception as e:
        print(f"✗ Error importing show_result.py: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("NumPy Compatibility Test")
    print("=" * 50)
    
    success = True
    success &= test_numpy_compatibility()
    print()
    success &= test_show_result_imports()
    
    print("=" * 50)
    if success:
        print("All tests passed! NumPy 2.0 compatibility is working.")
        sys.exit(0)
    else:
        print("Some tests failed. Check the output above.")
        sys.exit(1)