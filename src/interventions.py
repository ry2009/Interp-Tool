"""
Intervention utilities for Interp-Toolkit
Implements activation patching and conditional modifications
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Any, Callable
import functools

class InterventionEngine:
    def __init__(self, model: HookedTransformer):
        """Initialize with a loaded model"""
        self.model = model
        self.hooks = []
        
    def patch_activation(self, layer: int, component: str, patch_fn: Callable):
        """Apply a patch function to specific layer and component"""
        hook_name = f'blocks.{layer}.{component}'
        
        def hook_fn(activations, hook):
            return patch_fn(activations)
            
        hook = self.model.add_hook(hook_name, hook_fn)
        self.hooks.append(hook)
        return hook
        
    def zero_ablation(self, layer: int, component: str = 'hook_resid_post'):
        """Zero out activations at specified layer"""
        def zero_fn(activations):
            return torch.zeros_like(activations)
            
        return self.patch_activation(layer, component, zero_fn)
        
    def mean_ablation(self, layer: int, component: str = 'hook_resid_post', 
                     baseline_activations: torch.Tensor = None):
        """Replace activations with mean values"""
        def mean_fn(activations):
            if baseline_activations is not None:
                return torch.full_like(activations, baseline_activations.mean())
            return torch.full_like(activations, activations.mean())
            
        return self.patch_activation(layer, component, mean_fn)
        
    def conditional_patch(self, layer: int, component: str, 
                         condition_fn: Callable, patch_fn: Callable):
        """Apply patch only when condition is met"""
        def conditional_fn(activations):
            if condition_fn(activations):
                return patch_fn(activations)
            return activations
            
        return self.patch_activation(layer, component, conditional_fn)
        
    def regex_sink_mitigation(self, target_layers: List[int]):
        """Specific intervention for regex-sink patterns"""
        def mitigation_fn(activations):
            # Reduce extreme activations that might cause regex sinks
            threshold = activations.std() * 2
            mean_val = activations.mean()
            
            # Clip extreme values
            clipped = torch.clamp(activations, 
                                mean_val - threshold, 
                                mean_val + threshold)
            return clipped
            
        hooks = []
        for layer in target_layers:
            hook = self.patch_activation(layer, 'hook_resid_post', mitigation_fn)
            hooks.append(hook)
            
        return hooks
        
    def attention_head_ablation(self, layer: int, head: int):
        """Ablate specific attention head"""
        def head_ablation_fn(activations):
            # activations shape: [batch, pos, head, d_head]
            new_activations = activations.clone()
            new_activations[:, :, head, :] = 0
            return new_activations
            
        return self.patch_activation(layer, 'attn.hook_z', head_ablation_fn)
        
    def run_with_interventions(self, text: str):
        """Run model with all active interventions"""
        tokens = self.model.to_tokens(text)
        logits = self.model(tokens)
        return logits
        
    def clear_hooks(self):
        """Remove all intervention hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def evaluate_intervention_effect(self, original_text: str, intervention_fn: Callable):
        """Evaluate the effect of an intervention"""
        # Get baseline
        tokens = self.model.to_tokens(original_text)
        baseline_logits = self.model(tokens)
        
        # Apply intervention
        hook = intervention_fn()
        
        # Get modified output
        modified_logits = self.model(tokens)
        
        # Compute difference
        logit_diff = (modified_logits - baseline_logits).abs().mean()
        
        # Clean up
        if hasattr(hook, 'remove'):
            hook.remove()
        elif isinstance(hook, list):
            for h in hook:
                h.remove()
                
        return {
            'logit_difference': float(logit_diff),
            'baseline_shape': baseline_logits.shape,
            'modified_shape': modified_logits.shape
        }

def create_regex_test_cases():
    """Create test cases for regex-sink analysis"""
    return [
        "Match pattern: \\d+",
        "Find all: [a-zA-Z]+",
        "Regex: .*?",
        "Pattern: ^start.*end$",
        "Complex: (?:group1|group2)+"
    ]

def analyze_regex_sink_layers(model: HookedTransformer, test_cases: List[str]):
    """Analyze which layers are most affected by regex patterns"""
    engine = InterventionEngine(model)
    results = {}
    
    for layer in range(model.cfg.n_layers):
        layer_effects = []
        
        for text in test_cases:
            def intervention():
                return engine.zero_ablation(layer)
                
            effect = engine.evaluate_intervention_effect(text, intervention)
            layer_effects.append(effect['logit_difference'])
            
        results[layer] = {
            'mean_effect': float(np.mean(layer_effects)),
            'std_effect': float(np.std(layer_effects)),
            'effects': layer_effects
        }
        
    return results 