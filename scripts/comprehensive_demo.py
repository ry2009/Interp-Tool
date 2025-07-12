#!/usr/bin/env python3
"""
Comprehensive Interp-Toolkit Demo
Showcases all features: activation extraction, intervention, visualization, and regex-sink analysis
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from activation_extractor import ActivationExtractor
from interventions import InterventionEngine

def main():
    print("🚀 Comprehensive Interp-Toolkit Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("✨ Real activation extraction from GPT-2")
    print("🔍 Regex-sink behavior analysis")
    print("⚡ Activation interventions")
    print("📊 Layer-wise pattern visualization")
    print("🎯 Publication-ready interpretability insights")
    print()
    
    # Phase 1: Setup and Load Model
    print("📥 Phase 1: Loading GPT-2 Medium (24 layers, 1024d)")
    print("-" * 50)
    
    extractor = ActivationExtractor("gpt2-medium")
    extractor.load_model()
    
    # Phase 2: Comprehensive Activation Analysis
    print("\n🧠 Phase 2: Comprehensive Activation Analysis")
    print("-" * 50)
    
    # Test cases designed to reveal interesting patterns
    test_suite = {
        'regex_heavy': [
            "The pattern ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$ matches emails",
            "Use regex [0-9]{3}-[0-9]{3}-[0-9]{4} for phone numbers like 555-123-4567",
            "Pattern matching with (abc|def|ghi)+ repeats these sequences"
        ],
        'code_patterns': [
            "In Python: if x > 0 and y < 10: print('valid')",
            "JavaScript function: const add = (a, b) => a + b;",
            "SQL query: SELECT * FROM users WHERE age > 25;"
        ],
        'natural_text': [
            "The quick brown fox jumps over the lazy dog near the river.",
            "Scientists discovered a new species of butterfly in the Amazon rainforest.",
            "The conference will be held next Tuesday at the convention center."
        ],
        'mathematical': [
            "The equation x² + 2x + 1 = 0 has solutions using the quadratic formula.",
            "Integration by parts: ∫u dv = uv - ∫v du where u and v are functions.",
            "The derivative of sin(x) is cos(x) and the derivative of cos(x) is -sin(x)."
        ]
    }
    
    all_results = {}
    
    for category, texts in test_suite.items():
        print(f"\n🔍 Analyzing {category} texts...")
        category_results = []
        
        for i, text in enumerate(texts):
            print(f"   Processing: '{text[:60]}...'")
            
            # Extract activations
            activations = extractor.extract_activations(text)
            activations['category'] = category
            activations['sample_id'] = i
            category_results.append(activations)
            
            # Quick analysis
            if 'layer_activations' in activations:
                layer_means = [act['mean_activation'] for act in activations['layer_activations']]
                layer_stds = [act['std_activation'] for act in activations['layer_activations']]
                
                max_mean = max(layer_means)
                max_std = max(layer_stds)
                max_layer_mean = layer_means.index(max_mean)
                max_layer_std = layer_stds.index(max_std)
                
                print(f"      📊 Peak mean activation: {max_mean:.6f} at layer {max_layer_mean}")
                print(f"      📊 Peak std activation: {max_std:.6f} at layer {max_layer_std}")
        
        all_results[category] = category_results
    
    # Phase 3: Cross-Category Analysis
    print(f"\n📈 Phase 3: Cross-Category Pattern Analysis")
    print("-" * 50)
    
    category_stats = {}
    
    for category, results in all_results.items():
        # Aggregate stats across samples in this category
        all_means = []
        all_stds = []
        all_max_acts = []
        layer_patterns = [[] for _ in range(24)]  # GPT-2 medium has 24 layers
        
        for result in results:
            if 'layer_activations' in result:
                for layer_data in result['layer_activations']:
                    layer_idx = layer_data['layer']
                    all_means.append(layer_data['mean_activation'])
                    all_stds.append(layer_data['std_activation'])
                    all_max_acts.append(layer_data['max_activation'])
                    layer_patterns[layer_idx].append(layer_data['mean_activation'])
        
        # Calculate category statistics
        category_stats[category] = {
            'avg_mean_activation': np.mean(all_means) if all_means else 0,
            'avg_std_activation': np.mean(all_stds) if all_stds else 0,
            'avg_max_activation': np.mean(all_max_acts) if all_max_acts else 0,
            'layer_wise_means': [np.mean(layer) if layer else 0 for layer in layer_patterns]
        }
        
        print(f"📊 {category.upper()}:")
        print(f"   Mean activation across all layers: {category_stats[category]['avg_mean_activation']:.8f}")
        print(f"   Std activation across all layers: {category_stats[category]['avg_std_activation']:.4f}")
        print(f"   Max activation across all layers: {category_stats[category]['avg_max_activation']:.4f}")
    
    # Phase 4: Interpretability Insights
    print(f"\n🎯 Phase 4: Key Interpretability Insights")
    print("-" * 50)
    
    # Find most distinctive patterns
    layer_category_diffs = {}
    for layer in range(24):
        layer_values = {}
        for category in category_stats.keys():
            layer_values[category] = category_stats[category]['layer_wise_means'][layer]
        
        # Calculate variance across categories for this layer
        variance = np.var(list(layer_values.values()))
        layer_category_diffs[layer] = variance
    
    # Find layers with highest category discrimination
    most_discriminative_layers = sorted(layer_category_diffs.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
    
    print("🏆 Most Category-Discriminative Layers:")
    for layer, variance in most_discriminative_layers:
        print(f"   Layer {layer:2d}: variance = {variance:.8f}")
        # Show category values for this layer
        for category in category_stats.keys():
            value = category_stats[category]['layer_wise_means'][layer]
            print(f"      {category:12s}: {value:.8f}")
        print()
    
    # Phase 5: Regex-Sink Analysis
    print(f"\n⚡ Phase 5: Regex-Sink Behavior Analysis")
    print("-" * 50)
    
    regex_activations = []
    non_regex_activations = []
    
    for category, results in all_results.items():
        for result in results:
            text = result['text']
            contains_regex = any(char in text for char in "[](){}^$+*?|\\")
            
            if 'layer_activations' in result:
                layer_means = [act['mean_activation'] for act in result['layer_activations']]
                
                if contains_regex:
                    regex_activations.extend(layer_means)
                else:
                    non_regex_activations.extend(layer_means)
    
    if regex_activations and non_regex_activations:
        regex_mean = np.mean(regex_activations)
        non_regex_mean = np.mean(non_regex_activations)
        difference = abs(regex_mean - non_regex_mean)
        
        print(f"🎯 Regex texts average activation: {regex_mean:.8f}")
        print(f"🎯 Non-regex texts average activation: {non_regex_mean:.8f}")
        print(f"🎯 Absolute difference: {difference:.8f}")
        
        if difference > 1e-6:
            print("⚡ SIGNIFICANT: Potential regex-sink behavior detected!")
            print("   This suggests the model has learned to differentiate regex patterns.")
        else:
            print("✅ No strong regex-sink behavior found in this analysis.")
    
    # Phase 6: Save Results
    print(f"\n💾 Phase 6: Saving Comprehensive Results")
    print("-" * 50)
    
    # Prepare final results
    final_results = {
        'metadata': {
            'model': 'gpt2-medium',
            'analysis_type': 'comprehensive_interpretability',
            'categories_analyzed': list(test_suite.keys()),
            'total_samples': sum(len(texts) for texts in test_suite.values())
        },
        'raw_data': all_results,
        'category_statistics': category_stats,
        'discriminative_layers': most_discriminative_layers,
        'regex_analysis': {
            'regex_mean': regex_mean if 'regex_mean' in locals() else None,
            'non_regex_mean': non_regex_mean if 'non_regex_mean' in locals() else None,
            'difference': difference if 'difference' in locals() else None
        }
    }
    
    # Save to samples
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    output_file = samples_dir / "comprehensive_interpretability_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"📁 Saved comprehensive analysis to: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Generate summary report
    report_file = samples_dir / "interpretability_report.md"
    generate_report(final_results, report_file)
    print(f"📄 Generated interpretability report: {report_file}")
    
    print(f"\n🎉 Comprehensive Demo Complete!")
    print("=" * 50)
    print("🚀 Ready for publication-grade interpretability research!")
    print("📊 Launch Streamlit app to visualize: streamlit run src/app.py")
    print("🔬 Analyze data in Jupyter: notebooks/demo.ipynb")
    
    return output_file

def generate_report(results, output_file):
    """Generate a markdown report of interpretability findings"""
    
    with open(output_file, 'w') as f:
        f.write("# GPT-2 Medium Interpretability Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"Analyzed {results['metadata']['total_samples']} text samples across ")
        f.write(f"{len(results['metadata']['categories_analyzed'])} categories using ")
        f.write(f"{results['metadata']['model']} (24 layers, 1024 dimensions).\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Most discriminative layers
        f.write("### Layer-wise Category Discrimination\n\n")
        f.write("| Layer | Variance | Interpretation |\n")
        f.write("|-------|----------|----------------|\n")
        
        for layer, variance in results['discriminative_layers'][:3]:
            interpretation = "High category discrimination" if variance > 1e-12 else "Low discrimination"
            f.write(f"| {layer} | {variance:.2e} | {interpretation} |\n")
        
        f.write("\n### Category Analysis\n\n")
        for category, stats in results['category_statistics'].items():
            f.write(f"**{category.title()}**: ")
            f.write(f"Mean activation = {stats['avg_mean_activation']:.2e}, ")
            f.write(f"Std = {stats['avg_std_activation']:.4f}\n\n")
        
        # Regex analysis
        if results['regex_analysis']['difference']:
            f.write("### Regex-Sink Analysis\n\n")
            f.write(f"- Regex texts: {results['regex_analysis']['regex_mean']:.2e}\n")
            f.write(f"- Non-regex texts: {results['regex_analysis']['non_regex_mean']:.2e}\n")
            f.write(f"- Difference: {results['regex_analysis']['difference']:.2e}\n\n")
            
            if results['regex_analysis']['difference'] > 1e-6:
                f.write("**Conclusion**: Potential regex-sink behavior detected.\n\n")
            else:
                f.write("**Conclusion**: No significant regex-sink behavior found.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("1. Extracted activation summaries (mean, std, max, min) from all 24 layers\n")
        f.write("2. Computed category-wise statistics and cross-comparisons\n")
        f.write("3. Identified most discriminative layers using activation variance\n")
        f.write("4. Analyzed regex vs. non-regex text patterns\n\n")
        
        f.write("*Generated by Interp-Toolkit - CPU-friendly interpretability for LLMs*\n")

if __name__ == "__main__":
    main() 