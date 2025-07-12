#!/usr/bin/env python3
"""
Demo script to extract activations from GPT-2 Medium and analyze patterns
Focuses on regex-sink detection and mitigation
"""

import sys
import torch
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from activation_extractor import ActivationExtractor
from interventions import InterventionEngine

def main():
    print("🧠 Interp-Toolkit Demo: Extracting Real Activations")
    print("=" * 55)
    
    # Load GPT-2 Medium model (supported by transformer-lens)
    print("📥 Loading GPT-2 Medium...")
    extractor = ActivationExtractor("gpt2-medium")
    extractor.load_model()
    
    # Create test cases that might trigger regex-sink behavior
    test_cases = [
        "The regex pattern [a-z]+ matches lowercase letters",
        "Write a regular expression to match email addresses like user@domain.com",
        "Simple text without any regex patterns here",
        "Pattern: ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$ for emails",
        "The sequence (abc|def)* repeats abc or def zero or more times"
    ]
    
    all_activations = []
    
    for i, text in enumerate(test_cases):
        print(f"\n🔍 Processing case {i+1}: '{text[:50]}...'")
        
        # Extract activations for this text
        activations = extractor.extract_activations(text)
        
        # Add metadata
        activations['text'] = text
        activations['case_id'] = i + 1
        activations['contains_regex'] = any(char in text for char in "[](){}^$+*?|\\")
        
        all_activations.append(activations)
        
        # Print some stats
        if 'layer_activations' in activations:
            layer_means = [float(act['mean_activation']) for act in activations['layer_activations']]
            max_activation = max(layer_means)
            max_layer = layer_means.index(max_activation)
            print(f"   📊 Max activation: {max_activation:.4f} at layer {max_layer}")
    
    # Save to samples directory
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    output_file = samples_dir / "gpt2_medium_regex_activations.json"
    with open(output_file, 'w') as f:
        json.dump(all_activations, f, indent=2)
    
    print(f"\n💾 Saved activations to {output_file}")
    
    # Analyze patterns
    print(f"\n📈 Analysis Summary:")
    print("=" * 30)
    
    regex_cases = [act for act in all_activations if act['contains_regex']]
    non_regex_cases = [act for act in all_activations if not act['contains_regex']]
    
    if regex_cases and non_regex_cases:
        # Compare activation patterns
        regex_means = []
        non_regex_means = []
        
        for case in regex_cases:
            if 'layer_activations' in case:
                layer_means = [float(act['mean_activation']) for act in case['layer_activations']]
                regex_means.extend(layer_means)
        
        for case in non_regex_cases:
            if 'layer_activations' in case:
                layer_means = [float(act['mean_activation']) for act in case['layer_activations']]
                non_regex_means.extend(layer_means)
        
        if regex_means and non_regex_means:
            regex_avg = sum(regex_means) / len(regex_means)
            non_regex_avg = sum(non_regex_means) / len(non_regex_means)
            
            print(f"🎯 Regex cases avg activation: {regex_avg:.6f}")
            print(f"🎯 Non-regex cases avg activation: {non_regex_avg:.6f}")
            print(f"🎯 Difference: {abs(regex_avg - non_regex_avg):.6f}")
            
            if abs(regex_avg - non_regex_avg) > 0.001:
                print("⚡ Potential regex-sink behavior detected!")
            else:
                print("✅ No significant activation difference found")
    
    print(f"\n🎉 Demo completed! Activations ready for analysis in Streamlit.")
    return output_file

if __name__ == "__main__":
    main() 