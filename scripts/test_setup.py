#!/usr/bin/env python3
"""
Test script to verify Interp-Toolkit setup and basic functionality
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
        
    try:
        from transformer_lens import HookedTransformer
        print("✓ TransformerLens")
    except ImportError:
        print("✗ TransformerLens not found")
        return False
        
    try:
        import streamlit
        print("✓ Streamlit")
    except ImportError:
        print("✗ Streamlit not found")
        return False
        
    try:
        import plotly
        print("✓ Plotly")
    except ImportError:
        print("✗ Plotly not found")
        return False
        
    try:
        import pandas
        print("✓ Pandas")
    except ImportError:
        print("✗ Pandas not found")
        return False
        
    return True

def test_toolkit_imports():
    """Test that toolkit modules can be imported"""
    print("\nTesting toolkit modules...")
    
    try:
        from activation_extractor import ActivationExtractor
        print("✓ ActivationExtractor")
    except ImportError as e:
        print(f"✗ ActivationExtractor: {e}")
        return False
        
    try:
        from interventions import InterventionEngine
        print("✓ InterventionEngine")
    except ImportError as e:
        print(f"✗ InterventionEngine: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic functionality without downloading models"""
    print("\nTesting basic functionality...")
    
    try:
        from activation_extractor import ActivationExtractor
        extractor = ActivationExtractor("gpt2")  # Small model for testing
        print("✓ ActivationExtractor initialization")
    except Exception as e:
        print(f"✗ ActivationExtractor initialization: {e}")
        return False
        
    return True

def main():
    """Run all tests"""
    print("Interp-Toolkit Setup Test")
    print("=" * 30)
    
    success = True
    
    success &= test_imports()
    success &= test_toolkit_imports()
    success &= test_basic_functionality()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed! Setup is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check your installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 