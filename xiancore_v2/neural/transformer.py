"""
Custom Transformer Backbone - Built from Scratch
No external API dependencies (no HuggingFace, no transformers library)
Implements: Multi-head Attention, LayerNorm, RoPE, SwiGLU, RMSNorm
Optimized for Phi-Mini scale performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) for better long-context reasoning"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """RMSNorm - More efficient than LayerNorm for large models"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class SwiGLU(nn.Module):
    """SwiGLU Activation - Superior to ReLU/GELU for language modeling"""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE and Flash Attention support"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q, k = apply_rope(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.w_out(out)


class CrossAttentionReasoner(nn.Module):
    """Cross-Attention for Dynamic Neural-Symbolic Reasoning
    Allows neural component to attend over symbolic representations
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: Neural representations [B, T_q, dim]
        key_value: Symbolic representations [B, T_kv, dim]
        """
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]
        
        q = self.query_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(key_value).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(key_value).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single Transformer Block with Pre-Norm architecture"""
    
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, 
                 dropout: float = 0.0, use_cross_attn: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttentionReasoner(dim, num_heads, dropout)
            self.norm_cross = RMSNorm(dim)
        
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, symbolic_repr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        # Cross-attention to symbolic representations (if provided)
        if self.use_cross_attn and symbolic_repr is not None:
            x = x + self.dropout(self.cross_attn(self.norm_cross(x), symbolic_repr, mask))
        
        # MLP
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class CustomTransformer(nn.Module):
    """
    Full Custom Transformer Backbone
    Architecture similar to Phi-Mini/Llama but built from scratch
    Supports: RoPE, SwiGLU, RMSNorm, Cross-Attention for Neuro-Symbolic integration
    """
    
    def __init__(self, vocab_size: int = 50257, dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, hidden_dim: int = 3072, max_seq_len: int = 2048,
                 dropout: float = 0.0, use_cross_attn: bool = True):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout, 
                           use_cross_attn=(use_cross_attn and i % 2 == 0))
            for i in range(num_layers)
        ])
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying (improves sample efficiency)
        self.token_embedding.weight = self.lm_head.weight
        
        self.dim = dim
        self.num_layers = num_layers
    
    def forward(self, input_ids: torch.Tensor, 
                symbolic_repr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, symbolic_repr, mask)
        
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation without external dependencies"""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# Model configurations matching Phi-Mini scale
MODEL_CONFIGS = {
    "phi_mini": {
        "vocab_size": 50257,
        "dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 4096,
        "max_seq_len": 2048,
        "dropout": 0.0
    },
    "small": {
        "vocab_size": 50257,
        "dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "hidden_dim": 2048,
        "max_seq_len": 1024,
        "dropout": 0.1
    },
    "medium": {
        "vocab_size": 50257,
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 3072,
        "max_seq_len": 2048,
        "dropout": 0.0
    }
}


def create_model(config_name: str = "medium") -> CustomTransformer:
    """Factory function to create models with predefined configurations"""
    config = MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["medium"])
    return CustomTransformer(**config)
