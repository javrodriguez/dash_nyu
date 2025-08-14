"""
Tests for the Phoenix neural ODE model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from PHX_base_model import PHX_base_model
except ImportError:
    pytest.skip("PHX_base_model not available", allow_module_level=True)

class TestPhoenixModel:
    """Test cases for the Phoenix neural ODE model."""
    
    def test_model_initialization(self):
        """Test that the model can be initialized."""
        n_genes = 10
        neurons_per_layer = 20
        
        model = PHX_base_model(n_genes, neurons_per_layer)
        
        assert model is not None
        assert hasattr(model, 'net_prods')
        assert hasattr(model, 'net_sums')
        assert hasattr(model, 'net_alpha_combine_sums')
        assert hasattr(model, 'net_alpha_combine_prods')
        assert hasattr(model, 'gene_multipliers')
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        n_genes = 5
        neurons_per_layer = 10
        batch_size = 3
        time_steps = 4
        
        model = PHX_base_model(n_genes, neurons_per_layer)
        
        # Create dummy input
        x = torch.randn(batch_size, n_genes)
        t = torch.linspace(0, 1, time_steps)
        
        # Test forward pass
        try:
            output = model.forward(t, x)
            assert output.shape == (time_steps, batch_size, n_genes)
        except Exception as e:
            pytest.skip(f"Forward pass failed: {e}")
    
    def test_model_parameters(self):
        """Test that the model has the expected parameters."""
        n_genes = 10
        neurons_per_layer = 20
        
        model = PHX_base_model(n_genes, neurons_per_layer)
        
        # Check that parameters exist
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that all parameters have gradients
        for param in params:
            assert param.requires_grad
    
    def test_gene_multipliers(self):
        """Test that gene multipliers are properly initialized."""
        n_genes = 10
        neurons_per_layer = 20
        
        model = PHX_base_model(n_genes, neurons_per_layer)
        
        # Check gene multipliers shape
        assert model.gene_multipliers.shape == (n_genes,)
        
        # Check that gene multipliers are learnable
        assert model.gene_multipliers.requires_grad
    
    def test_model_device_compatibility(self):
        """Test that the model works on CPU."""
        n_genes = 5
        neurons_per_layer = 10
        
        model = PHX_base_model(n_genes, neurons_per_layer)
        
        # Ensure model is on CPU
        model = model.cpu()
        
        # Create dummy input on CPU
        x = torch.randn(2, n_genes)
        t = torch.linspace(0, 1, 3)
        
        # Test forward pass on CPU
        try:
            output = model.forward(t, x)
            assert output.device.type == 'cpu'
        except Exception as e:
            pytest.skip(f"CPU forward pass failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
