"""
Differentiable Neural Logic Machines
Transforms rigid rule-based logic into trainable differentiable operations
Supports: AND, OR, NOT, IMPLIES with learnable parameters via backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DifferentiableLogicGate(nn.Module):
    """Base class for differentiable logic gates with learnable parameters"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DifferentiableAND(DifferentiableLogicGate):
    """Differentiable AND gate using soft minimum (t-norm)"""
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Soft minimum using temperature-controlled softmax
        stacked = torch.stack([a, b], dim=-1)
        return -self.temperature * torch.logsumexp(-stacked / self.temperature, dim=-1)


class DifferentiableOR(DifferentiableLogicGate):
    """Differentiable OR gate using soft maximum (t-conorm)"""
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Soft maximum using temperature-controlled softmax
        stacked = torch.stack([a, b], dim=-1)
        return self.temperature * torch.logsumexp(stacked / self.temperature, dim=-1)


class DifferentiableNOT(DifferentiableLogicGate):
    """Differentiable NOT gate (standard fuzzy negation)"""
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return 1.0 - a


class DifferentiableIMPLIES(DifferentiableLogicGate):
    """Differentiable IMPLIES: A → B ≡ ¬A ∨ B"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature)
        self.not_gate = DifferentiableNOT()
        self.or_gate = DifferentiableOR(temperature)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        not_a = self.not_gate(a)
        return self.or_gate(not_a, b)


class NeuralLogicLayer(nn.Module):
    """
    Neural Logic Layer - Combines multiple differentiable gates
    Learns logical rules through backpropagation
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_rules: int = 8,
                 hidden_dim: int = 64, temperature: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rules = num_rules
        
        # Learnable rule weights
        self.rule_weights = nn.Parameter(torch.randn(num_rules, input_dim))
        self.rule_bias = nn.Parameter(torch.zeros(num_rules))
        
        # Gate parameters (learnable temperature)
        self.and_temp = nn.Parameter(torch.tensor(temperature))
        self.or_temp = nn.Parameter(torch.tensor(temperature))
        
        # Output projection - aggregated is [batch, 1], so input is 1
        self.output_proj = nn.Linear(1, output_dim)
        
        # Attention over rules (dynamic rule selection)
        self.rule_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor [batch_size, input_dim] with values in [0, 1] (truth values)
        Returns: Output tensor [batch_size, output_dim] with learned logical transformations
        """
        batch_size = x.shape[0]
        
        # Compute rule activations using differentiable logic
        rule_inputs = torch.sigmoid(x @ self.rule_weights.T + self.rule_bias)  # [batch, num_rules]
        
        # Apply differentiable AND/OR combinations
        and_gate = DifferentiableAND(self.and_temp.clamp(0.1, 2.0))
        or_gate = DifferentiableOR(self.or_temp.clamp(0.1, 2.0))
        
        # Pairwise AND operations
        and_results = []
        for i in range(0, self.num_rules - 1, 2):
            if i + 1 < self.num_rules:
                and_result = and_gate(rule_inputs[:, i], rule_inputs[:, i + 1])
                and_results.append(and_result)
        
        # Combine with OR
        if len(and_results) >= 2:
            combined = torch.stack(and_results, dim=-1)
            or_result = or_gate(combined[..., 0], combined[..., 1])
        elif len(and_results) == 1:
            or_result = and_results[0]
        else:
            or_result = rule_inputs.mean(dim=-1)
        
        # Dynamic rule weighting via attention
        attention_weights = self.rule_attention(x)  # [batch_size, num_rules]
        weighted_rules = rule_inputs * attention_weights
        
        # Aggregate rules - sum across rules dimension
        aggregated = weighted_rules.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Final output
        output = self.output_proj(aggregated)  # [batch_size, output_dim]
        return torch.sigmoid(output)


class DifferentiableLogicMachine(nn.Module):
    """
    Full Differentiable Logic Machine
    Multi-layer architecture for complex logical reasoning
    Supports: Multi-hop reasoning, rule learning, confidence propagation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, num_rules_per_layer: int = 16,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            NeuralLogicLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_rules=num_rules_per_layer,
                hidden_dim=hidden_dim // 2,
                temperature=0.5 + i * 0.1  # Increasing temperature for deeper layers
            )
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Confidence tracking
        self.confidence_tracker = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, facts: torch.Tensor, 
                rules: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        facts: Input facts as truth values [batch_size, input_dim]
        rules: Optional rule priors [batch_size, num_rules]
        
        Returns:
            conclusions: Derived conclusions [batch_size, output_dim]
            confidence: Confidence scores [batch_size, 1]
        """
        # Project input to hidden dimension
        x = torch.sigmoid(self.input_projection(facts))
        
        # Pass through logic layers
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        
        # Generate conclusions
        conclusions = torch.sigmoid(self.output_layer(x))
        
        # Compute confidence
        confidence = self.confidence_tracker(x)
        
        return conclusions, confidence
    
    def extract_learned_rules(self, threshold: float = 0.7) -> List[dict]:
        """Extract human-readable rules from learned parameters"""
        rules = []
        for i, layer in enumerate(self.layers):
            rule_weights = layer.rule_weights.detach()
            for rule_idx in range(self.num_rules):
                weights = rule_weights[rule_idx]
                significant_vars = torch.where(torch.abs(weights) > threshold)[0]
                if len(significant_vars) > 0:
                    rules.append({
                        'layer': i,
                        'rule_id': rule_idx,
                        'variables': significant_vars.tolist(),
                        'weights': weights[significant_vars].tolist(),
                        'bias': layer.rule_bias[rule_idx].item()
                    })
        return rules


class NeuroSymbolicIntegrator(nn.Module):
    """
    Integrates Neural Logic Machine with Transformer representations
    Enables end-to-end training of neuro-symbolic reasoning
    """
    
    def __init__(self, transformer_dim: int, logic_input_dim: int,
                 hidden_dim: int = 128, num_logic_layers: int = 3):
        super().__init__()
        
        # Project transformer outputs to logic input
        self.neural_to_logic = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, logic_input_dim),
            nn.Sigmoid()  # Convert to truth values
        )
        
        # Differentiable logic machine
        self.logic_machine = DifferentiableLogicMachine(
            input_dim=logic_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_logic_layers,
            output_dim=logic_input_dim
        )
        
        # Project logic outputs back to neural space
        self.logic_to_neural = nn.Linear(logic_input_dim, transformer_dim)
        
        # Gating mechanism (decides how much to trust logic vs neural)
        self.gate = nn.Sequential(
            nn.Linear(transformer_dim + logic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, neural_repr: torch.Tensor, 
                symbolic_facts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        neural_repr: Transformer representations [batch_size, seq_len, transformer_dim]
        symbolic_facts: Optional external symbolic facts
        
        Returns:
            integrated_repr: Combined neuro-symbolic representation
            logic_output: Pure logic output
            gate_values: Trust scores for logic component
        """
        batch_size, seq_len, _ = neural_repr.shape
        
        # Convert neural to logic space
        logic_input = self.neural_to_logic(neural_repr.view(-1, neural_repr.shape[-1]))
        logic_input = logic_input.view(batch_size, seq_len, -1)
        
        # Apply symbolic facts if provided
        if symbolic_facts is not None:
            logic_input = 0.5 * logic_input + 0.5 * symbolic_facts
        
        # Apply differentiable logic
        logic_output, confidence = self.logic_machine(
            logic_input.view(-1, logic_input.shape[-1])
        )
        logic_output = logic_output.view(batch_size, seq_len, -1)
        confidence = confidence.view(batch_size, seq_len, -1)
        
        # Project back to neural space
        neural_from_logic = self.logic_to_neural(logic_output)
        
        # Gating mechanism
        combined_input = torch.cat([neural_repr, neural_from_logic], dim=-1)
        gate_values = self.gate(combined_input.view(-1, combined_input.shape[-1]))
        gate_values = gate_values.view(batch_size, seq_len, -1)
        
        # Integrated representation
        integrated_repr = (1 - gate_values) * neural_repr + gate_values * neural_from_logic
        
        return integrated_repr, logic_output, gate_values
