"""
XianCore V2 - Neuro-Symbolic AI System
Production-Ready Architecture with:
- Custom Transformer Backbone (no external API dependency)
- Differentiable Neural Logic Machines
- FAISS-based Vector Storage for billion-scale facts
- Cross-Attention Dynamic Reasoning
- Multi-Agent Debate with Fact-Checking
- 4-bit/8-bit Quantization + LoRA Support
- Parallel Document Ingestion Pipeline
"""

__version__ = "2.0.0"
__author__ = "XianCore Team"

from .core.engine import XianCoreEngine, create_xiancore
from .neural.transformer import CustomTransformer, CrossAttentionReasoner, create_model
from .symbolic.logic import DifferentiableLogicMachine, NeuralLogicLayer
from .storage.vector_db import FAISSVectorStore, BillionScaleIndex
from .agents.debate import MultiAgentDebateSystem, FactChecker
from .utils.quantization import QuantizationManager, LoRALinear, apply_lora_to_model

__all__ = [
    "XianCoreEngine",
    "create_xiancore",
    "CustomTransformer",
    "CrossAttentionReasoner",
    "create_model",
    "DifferentiableLogicMachine",
    "NeuralLogicLayer",
    "FAISSVectorStore",
    "BillionScaleIndex",
    "MultiAgentDebateSystem",
    "FactChecker",
    "QuantizationManager",
    "LoRALinear",
    "apply_lora_to_model",
]
