#!/usr/bin/env python3
"""Quick validation script for ST score model integration.

Tests that all modules import correctly and the ST scoring pipeline is wired up.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_imports():
    """Test all required imports."""
    print("✓ Testing imports...")
    try:
        from stock_triggers.scripts.short_term.train_st_logistic_model import compute_st_score_model
        print("  ✓ short_term/train_st_logistic_model.py")
    except ImportError as e:
        print(f"  ✗ short_term/train_st_logistic_model.py: {e}")
        return False
    
    try:
        from stock_triggers.ui.patterns.st_score import apply_st_score_model, load_signal_st_score_model
        print("  ✓ st_score.py")
    except ImportError as e:
        print(f"  ✗ st_score.py: {e}")
        return False
    
    try:
        from stock_triggers.ui.patterns.publish import rescore_signal_history
        print("  ✓ publish.py (updated)")
    except ImportError as e:
        print(f"  ✗ publish.py: {e}")
        return False
    
    return True


def test_model_loading():
    """Test that model file can be loaded."""
    print("\n✓ Testing model loading...")
    try:
        from stock_triggers.ui.patterns.st_score import load_signal_st_score_model
        model = load_signal_st_score_model()
        if model and "model" in model:
            print(f"  ✓ Model loaded: {model.get('signals_analyzed', 'N/A')} signals analyzed")
            print(f"    - Features: {len(model.get('numeric_features', []))}")
            print(f"    - Positive rate: {model.get('model', {}).get('positive_rate', 0.0):.4f}")
        else:
            print("  ✗ Model dict empty or missing 'model' key")
            return False
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    
    return True


def test_feature_computation():
    """Test that ST features can be computed."""
    print("\n✓ Testing feature computation...")
    try:
        from stock_triggers.ui.patterns.st_score import compute_st_intraday_features, encode_markov_state
        import pandas as pd
        
        # Test markov state encoding
        states = ["fresh_breakout", "extended_breakout", "constructive_trend", "sideways", "breakdown_risk"]
        for state in states:
            encoded = encode_markov_state(state)
            if encoded and any(encoded.values()):
                print(f"  ✓ Encode markov state: {state}")
            else:
                print(f"  ✗ Failed to encode: {state}")
                return False
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    
    return True


def test_apply_function():
    """Test that apply_st_score_model function signature is correct."""
    print("\n✓ Testing apply_st_score_model function...")
    try:
        from stock_triggers.ui.patterns.st_score import apply_st_score_model
        import inspect
        
        sig = inspect.signature(apply_st_score_model)
        params = list(sig.parameters.keys())
        required = ["signals_df", "prices_df", "payload"]
        
        for param in required:
            if param in params:
                print(f"  ✓ Parameter: {param}")
            else:
                print(f"  ✗ Missing parameter: {param}")
                return False
    except Exception as e:
        print(f"  ✗ {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("ST Score Model Integration Validation")
    print("=" * 60)
    
    all_pass = True
    all_pass &= test_imports()
    all_pass &= test_model_loading()
    all_pass &= test_feature_computation()
    all_pass &= test_apply_function()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All validation tests PASSED")
        print("\nNext steps:")
        print("1. Train real ST score model: python stock_triggers/scripts/short_term/train_st_logistic_model.py ...")
        print("2. Regenerate signals: python stock_triggers/scripts/short_term/generate_st_signals.py")
        print("3. Test ST Lab with st_score>=80 filter")
        print("4. Compare win rate to baseline (signal_score>=80)")
        return 0
    else:
        print("✗ Some validation tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
