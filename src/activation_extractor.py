"""
Activation Extractor for Interp-Toolkit
Extracts and saves activation dumps from LLMs in JSON format
"""

import torch
import json
from transformer_lens import HookedTransformer
from typing import Dict, List, Any
import numpy as np

class ActivationExtractor:
    def __init__(self, model_name: str):
        """Initialize with model name"""
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the specified model"""
        try:
            self.model = HookedTransformer.from_pretrained(self.model_name)
            print(f"Loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def extract_activations(self, text: str, layers: List[int] = None) -> Dict[str, Any]:
        """Extract activations for given text"""
        if self.model is None:
            self.load_model()
            
        tokens = self.model.to_tokens(text)
        logits, cache = self.model.run_with_cache(tokens)
        
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))
            
        activations = {
            'text': text,
            'model': self.model_name,
            'layers': {}
        }
        
        for layer in layers:
            # Extract residual stream activations
            resid_post = cache[f'blocks.{layer}.hook_resid_post']
            attn_out = cache[f'blocks.{layer}.attn.hook_z']
            mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
            
            activations['layers'][layer] = {
                'resid_post': resid_post.detach().cpu().numpy().tolist(),
                'attn_out': attn_out.detach().cpu().numpy().tolist(),
                'mlp_out': mlp_out.detach().cpu().numpy().tolist(),
                'mean_activation': float(resid_post.mean().item())
            }
            
        return activations
    
    def save_activations(self, activations: Dict[str, Any], filename: str):
        """Save activations to JSON file"""
        with open(filename, 'w') as f:
            json.dump(activations, f, indent=2)
        print(f"Saved activations to {filename}")
        
    def extract_regex_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Extract activations for regex pattern analysis"""
        all_activations = []
        
        for text in texts:
            activations = self.extract_activations(text)
            all_activations.append(activations)
            
        return {
            'regex_analysis': True,
            'samples': all_activations,
            'summary': self._compute_summary(all_activations)
        }
        
    def _compute_summary(self, activations_list: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics"""
        layer_means = {}
        
        for activations in activations_list:
            for layer, data in activations['layers'].items():
                if layer not in layer_means:
                    layer_means[layer] = []
                layer_means[layer].append(data['mean_activation'])
                
        summary = {}
        for layer, means in layer_means.items():
            summary[layer] = {
                'mean': float(np.mean(means)),
                'std': float(np.std(means)),
                'min': float(np.min(means)),
                'max': float(np.max(means))
            }
            
        return summary

# Convenience functions for specific models
def extract_tinyllama_activations(text: str, save_path: str = None):
    """Extract activations from TinyLlama model"""
    extractor = ActivationExtractor('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    activations = extractor.extract_activations(text)
    
    if save_path:
        extractor.save_activations(activations, save_path)
        
    return activations

def extract_gemma_activations(text: str, save_path: str = None):
    """Extract activations from Gemma model"""
    extractor = ActivationExtractor('google/gemma-2b-it')
    activations = extractor.extract_activations(text)
    
    if save_path:
        extractor.save_activations(activations, save_path)
        
    return activations 