"""
Quantization and LoRA Utilities for Efficient Deployment
Supports: 4-bit/8-bit quantization, Low-Rank Adaptation (LoRA), 
gradient checkpointing, memory-efficient training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    bits: int = 8  # 4 or 8
    symmetric: bool = True
    per_channel: bool = False
    dynamic: bool = True
    quantize_embeddings: bool = False
    quantize_attention: bool = True


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer
    Supports 4-bit and 8-bit integer quantization
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, config: Optional[QuantizationConfig] = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Full precision weight (used during training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantized weight (cached)
        self.register_buffer('weight_quantized', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        
        self._quantize_weight()
    
    def _quantize_weight(self):
        """Quantize weights to specified bit width"""
        if self.config.bits == 8:
            qmin, qmax = -128, 127
        elif self.config.bits == 4:
            qmin, qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bit width: {self.config.bits}")
        
        weight = self.weight
        
        if self.config.symmetric:
            # Symmetric quantization
            scale = weight.abs().max() / float(qmax)
            zero_point = torch.tensor(0.0)
        else:
            # Asymmetric quantization
            scale = (weight.max() - weight.min()) / float(qmax - qmin)
            zero_point = (-weight.min() / scale).round().clamp(qmin, qmax)
        
        # Quantize
        weight_int = (weight / scale + zero_point).round().clamp(qmin, qmax)
        
        # Dequantize for forward pass (simulated quantization)
        self.weight_quantized = (weight_int - zero_point) * scale
        self.scale = scale
        self.zero_point = zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-quantize if in training mode (for gradient flow)
        if self.training:
            self._quantize_weight()
        
        return F.linear(x, self.weight_quantized, self.bias)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Layer
    Efficient fine-tuning with minimal parameters
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: float = 16.0, 
                 dropout: float = 0.05, merge_weights: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # LoRA matrices (low-rank decomposition)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout)
        
        # Merged flag
        self.merged = False
    
    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation to base linear transformation
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            base_weight: Base weight matrix [out_features, in_features]
            
        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        if self.merged and self.merge_weights:
            # Use merged weights
            result = F.linear(x, base_weight)
        else:
            # Base transformation
            result = F.linear(x, base_weight)
            
            # LoRA adaptation
            lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_output * self.scaling
        
        return result


class LoRALinear(nn.Module):
    """
    Complete LoRA-enabled Linear Layer
    Wraps standard linear with LoRA adaptation
    """
    
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0, 
                 dropout: float = 0.05, bias: bool = True):
        super().__init__()
        
        # Base linear layer
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adapter
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x, self.base_linear.weight)
    
    def merge_and_save(self):
        """Merge LoRA weights into base weights for inference"""
        if not self.lora.merged:
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_linear.weight.data += delta_w
            self.lora.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights (for continued training)"""
        if self.lora.merged:
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_linear.weight.data -= delta_w
            self.lora.merged = False


def apply_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                        target_modules: List[str] = None) -> nn.Module:
    """
    Apply LoRA to specified modules in a model
    
    Args:
        model: PyTorch model
        rank: LoRA rank
        alpha: LoRA alpha parameter
        target_modules: List of module names to apply LoRA (e.g., ['w_q', 'w_k', 'w_v'])
        
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ['w_q', 'w_k', 'w_v', 'w_out']
    
    for name, module in model.named_modules():
        parts = name.split('.')
        if parts[-1] in target_modules and isinstance(module, nn.Linear):
            parent_name = '.'.join(parts[:-1])
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create LoRA linear
            lora_linear = LoRALinear(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha
            )
            
            # Copy weights
            lora_linear.base_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_linear.base_linear.bias.data = module.bias.data.clone()
            
            # Replace module
            setattr(parent, parts[-1], lora_linear)
    
    return model


class QuantizationManager:
    """
    Manages quantization for entire models
    Handles conversion, calibration, and inference
    """
    
    def __init__(self, model: nn.Module, config: Optional[QuantizationConfig] = None):
        self.model = model
        self.config = config or QuantizationConfig()
        self.original_state = None
    
    def prepare_for_quantization(self):
        """Save original state and prepare model"""
        self.original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def quantize_model(self, exclude_modules: List[str] = None) -> nn.Module:
        """
        Convert model to quantized version
        
        Args:
            exclude_modules: List of module types to exclude from quantization
            
        Returns:
            Quantized model
        """
        if exclude_modules is None:
            exclude_modules = ['LayerNorm', 'RMSNorm', 'Embedding']
        
        for name, module in self.model.named_modules():
            # Skip excluded modules
            if any(excl in type(module).__name__ for excl in exclude_modules):
                continue
            
            # Replace linear layers
            if isinstance(module, nn.Linear):
                quantized = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    config=self.config
                )
                quantized.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized.bias.data = module.bias.data.clone()
                
                # Find parent and replace
                parts = name.split('.')
                parent_name = '.'.join(parts[:-1])
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, parts[-1], quantized)
        
        return self.model
    
    def calibrate(self, calibration_data: torch.Tensor, num_batches: int = 100):
        """
        Calibrate quantization parameters using representative data
        
        Args:
            calibration_data: Representative input data
            num_batches: Number of batches for calibration
        """
        self.model.eval()
        
        with torch.no_grad():
            for i in range(min(num_batches, len(calibration_data))):
                batch = calibration_data[i:i+1]
                _ = self.model(batch)
    
    def export_quantized(self, path: str):
        """Export quantized model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    def load_quantized(self, path: str):
        """Load quantized model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory-efficient training
    Trades compute for memory
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        # Manual implementation for custom models
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True


def get_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """
    Calculate memory footprint of model components
    
    Returns:
        Dictionary with memory usage in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_memory = total_params * 4 / 1024 / 1024  # FP32 = 4 bytes
    
    # Estimate activation memory (rough approximation)
    activation_memory = param_memory * 0.5  # Typically 30-50% of param memory
    
    return {
        'total_parameters_millions': total_params / 1e6,
        'trainable_parameters_millions': trainable_params / 1e6,
        'parameter_memory_mb': param_memory,
        'estimated_activation_memory_mb': activation_memory,
        'total_estimated_memory_mb': param_memory + activation_memory
    }


def create_efficient_config(model_size: str = "medium") -> Dict:
    """
    Create efficient training configuration based on model size
    
    Args:
        model_size: "small", "medium", "large"
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "small": {
            "quantization_bits": 8,
            "lora_rank": 8,
            "lora_alpha": 16,
            "gradient_checkpointing": False,
            "mixed_precision": True
        },
        "medium": {
            "quantization_bits": 8,
            "lora_rank": 16,
            "lora_alpha": 32,
            "gradient_checkpointing": True,
            "mixed_precision": True
        },
        "large": {
            "quantization_bits": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "gradient_checkpointing": True,
            "mixed_precision": True
        }
    }
    
    return configs.get(model_size, configs["medium"])
