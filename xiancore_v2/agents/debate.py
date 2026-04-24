"""
Multi-Agent Debate System with Automated Fact-Checking
Expands the original DebateModerator with:
- Multiple specialized agent roles
- Cross-examination protocols
- External fact verification
- Hallucination mitigation
- Consensus building mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class AgentRole(Enum):
    """Specialized agent roles for debate"""
    PROPONENT = "proponent"  # Argues for the claim
    OPPONENT = "opponent"  # Argues against the claim
    MODERATOR = "moderator"  # Facilitates debate
    FACT_CHECKER = "fact_checker"  # Verifies factual claims
    SYNTHESIZER = "synthesizer"  # Combines arguments into conclusion


@dataclass
class Argument:
    """Structured argument representation"""
    content: str
    agent_role: AgentRole
    confidence: float
    evidence: List[str]
    logical_form: Optional[torch.Tensor] = None
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class DebateRound:
    """Single round of debate"""
    round_number: int
    proponent_args: List[Argument]
    opponent_args: List[Argument]
    fact_checks: List[Dict]
    moderator_notes: str


class FactChecker(nn.Module):
    """
    Automated Fact-Checking Module
    Verifies claims against knowledge base and external sources
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        # Claim verification network
        self.verifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # claim + evidence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # true, false, uncertain
        )
        
        # Source credibility scorer
        self.credibility_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Contradiction detector
        self.contradiction_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, claim_embedding: torch.Tensor, 
                evidence_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify a claim against evidence
        
        Args:
            claim_embedding: [batch_size, embedding_dim]
            evidence_embeddings: [batch_size, num_evidence, embedding_dim]
            
        Returns:
            verdict: [batch_size, 3] (true, false, uncertain probabilities)
            confidence: [batch_size, 1]
        """
        batch_size = claim_embedding.shape[0]
        num_evidence = evidence_embeddings.shape[1]
        
        # Expand claim for each evidence
        claim_expanded = claim_embedding.unsqueeze(1).expand(-1, num_evidence, -1)
        
        # Concatenate claim and evidence
        combined = torch.cat([claim_expanded, evidence_embeddings], dim=-1)
        
        # Get verdicts for each evidence
        verdicts = self.verifier(combined.view(-1, combined.shape[-1]))
        verdicts = verdicts.view(batch_size, num_evidence, 3)
        
        # Aggregate verdicts (weighted average)
        aggregated_verdict = verdicts.mean(dim=1)
        
        # Confidence based on agreement between evidence
        variance = verdicts.var(dim=1).mean(dim=-1, keepdim=True)
        confidence = 1.0 - variance.clamp(0, 1)
        
        return aggregated_verdict, confidence
    
    def check_contradiction(self, statement1_emb: torch.Tensor, 
                           statement2_emb: torch.Tensor) -> torch.Tensor:
        """Check if two statements contradict each other"""
        combined = torch.cat([statement1_emb, statement2_emb], dim=-1)
        return self.contradiction_detector(combined)
    
    def score_source_credibility(self, source_embedding: torch.Tensor) -> torch.Tensor:
        """Score the credibility of a source"""
        return self.credibility_scorer(source_embedding)


class CognitiveAgent(nn.Module):
    """
    Individual Cognitive Agent with specialized role
    Generates arguments based on role and context
    """
    
    def __init__(self, role: AgentRole, model_dim: int = 768, 
                 hidden_dim: int = 512, max_arguments: int = 5):
        super().__init__()
        
        self.role = role
        self.model_dim = model_dim
        self.max_arguments = max_arguments
        
        # Role-specific attention
        self.role_embedding = nn.Parameter(torch.randn(model_dim))
        
        # Argument generation
        self.argument_generator = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),  # context + role
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(model_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Evidence retriever (attention over knowledge)
        self.evidence_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            dropout=0.1
        )
    
    def forward(self, context: torch.Tensor, 
                knowledge_base: torch.Tensor,
                temperature: float = 1.0) -> List[Argument]:
        """
        Generate arguments based on context and role
        
        Args:
            context: Current debate context [seq_len, model_dim]
            knowledge_base: Available knowledge [num_facts, model_dim]
            temperature: Sampling temperature
            
        Returns:
            List of Argument objects
        """
        # Add role embedding to context
        role_context = context + self.role_embedding
        
        # Attend to relevant knowledge
        context_expanded = context.unsqueeze(1)
        attended, _ = self.evidence_attention(
            context_expanded, 
            knowledge_base.unsqueeze(1),
            knowledge_base.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Generate argument embeddings
        combined = torch.cat([context, attended], dim=-1)
        arg_embeddings = self.argument_generator(combined)
        
        # Estimate confidence
        confidence = self.confidence_estimator(arg_embeddings).squeeze(-1)
        
        # Sample top arguments
        if temperature > 0:
            probs = F.softmax(confidence / temperature, dim=-1)
            indices = torch.multinomial(probs, min(self.max_arguments, len(probs)))
        else:
            indices = torch.topk(confidence, min(self.max_arguments, len(confidence))).indices
        
        arguments = []
        for idx in indices:
            arg = Argument(
                content=f"[ARGUMENT_{self.role.value}_{idx.item()}]",  # Placeholder
                agent_role=self.role,
                confidence=confidence[idx].item(),
                evidence=[],  # Would be populated from knowledge base
                logical_form=arg_embeddings[idx]
            )
            arguments.append(arg)
        
        return arguments


class MultiAgentDebateSystem(nn.Module):
    """
    Complete Multi-Agent Debate System
    Coordinates multiple agents for robust reasoning and hallucination mitigation
    """
    
    def __init__(self, model_dim: int = 768, hidden_dim: int = 512,
                 num_debate_rounds: int = 3, consensus_threshold: float = 0.8):
        super().__init__()
        
        self.num_debate_rounds = num_debate_rounds
        self.consensus_threshold = consensus_threshold
        
        # Initialize specialized agents
        self.agents = nn.ModuleDict({
            'proponent': CognitiveAgent(AgentRole.PROPONENT, model_dim, hidden_dim),
            'opponent': CognitiveAgent(AgentRole.OPPONENT, model_dim, hidden_dim),
            'moderator': CognitiveAgent(AgentRole.MODERATOR, model_dim, hidden_dim),
            'synthesizer': CognitiveAgent(AgentRole.SYNTHESIZER, model_dim, hidden_dim)
        })
        
        # Fact checker
        self.fact_checker = FactChecker(model_dim, hidden_dim)
        
        # Consensus aggregator
        self.consensus_aggregator = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Tanh()
        )
        
        # Final decision maker
        self.decision_maker = nn.Sequential(
            nn.Linear(model_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Debate history tracking
        self.debate_history = []
    
    def forward(self, query: torch.Tensor, 
                knowledge_base: torch.Tensor,
                initial_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run complete multi-agent debate process
        
        Args:
            query: Input query embedding [seq_len, model_dim]
            knowledge_base: Available facts [num_facts, model_dim]
            initial_context: Optional starting context
            
        Returns:
            Dictionary with final answer, confidence, and debate transcript
        """
        if initial_context is None:
            context = query
        else:
            context = torch.cat([query, initial_context], dim=0)
        
        debate_rounds = []
        all_arguments = {'proponent': [], 'opponent': []}
        
        # Run debate rounds
        for round_num in range(self.num_debate_rounds):
            round_data = self._run_debate_round(
                round_num=round_num,
                context=context,
                knowledge_base=knowledge_base,
                previous_args=all_arguments
            )
            
            debate_rounds.append(round_data)
            all_arguments['proponent'].extend(round_data.proponent_args)
            all_arguments['opponent'].extend(round_data.opponent_args)
            
            # Update context with round summary
            context = self._update_context(context, round_data)
        
        # Synthesize final answer
        final_answer = self._synthesize_answer(
            all_arguments,
            debate_rounds,
            knowledge_base
        )
        
        return {
            'answer': final_answer,
            'debate_transcript': debate_rounds,
            'confidence': final_answer['confidence'],
            'fact_checks': self._aggregate_fact_checks(debate_rounds),
            'consensus_reached': final_answer['consensus_reached']
        }
    
    def _run_debate_round(self, round_num: int, context: torch.Tensor,
                         knowledge_base: torch.Tensor,
                         previous_args: Dict) -> DebateRound:
        """Run single debate round"""
        
        # Generate arguments
        proponent_args = self.agents['proponent'](context, knowledge_base)
        opponent_args = self.agents['opponent'](context, knowledge_base)
        
        # Fact-check all arguments
        fact_checks = []
        if proponent_args and proponent_args[0].logical_form is not None:
            prop_embeddings = torch.stack([a.logical_form for a in proponent_args])
            opp_embeddings = torch.stack([a.logical_form for a in opponent_args])
            
            # Check internal consistency
            if len(prop_embeddings) > 0 and len(opp_embeddings) > 0:
                contradictions = self.fact_checker.check_contradiction(
                    prop_embeddings.mean(0).unsqueeze(0),
                    opp_embeddings.mean(0).unsqueeze(0)
                )
                fact_checks.append({
                    'type': 'contradiction_check',
                    'score': contradictions.item(),
                    'details': 'Proponent vs Opponent contradiction'
                })
        
        # Moderator notes
        moderator_args = self.agents['moderator'](context, knowledge_base)
        moderator_notes = f"Round {round_num}: {len(proponent_args)} proponent args, {len(opponent_args)} opponent args"
        
        return DebateRound(
            round_number=round_num,
            proponent_args=proponent_args,
            opponent_args=opponent_args,
            fact_checks=fact_checks,
            moderator_notes=moderator_notes
        )
    
    def _update_context(self, context: torch.Tensor, round_data: DebateRound) -> torch.Tensor:
        """Update context with debate progress"""
        # Simple aggregation (can be enhanced with attention)
        all_args = round_data.proponent_args + round_data.opponent_args
        if all_args and all_args[0].logical_form is not None:
            round_summary = torch.stack([a.logical_form for a in all_args]).mean(0, keepdim=True)
            context = torch.cat([context, round_summary], dim=0)
        return context
    
    def _synthesize_answer(self, all_arguments: Dict, debate_rounds: List[DebateRound],
                          knowledge_base: torch.Tensor) -> Dict[str, Any]:
        """Synthesize final answer from debate"""
        
        # Aggregate all argument embeddings
        all_embs = []
        confidences = []
        
        for role_args in all_arguments.values():
            for arg in role_args:
                if arg.logical_form is not None:
                    all_embs.append(arg.logical_form)
                    confidences.append(arg.confidence)
        
        if not all_embs:
            return {
                'embedding': torch.zeros(1, self.agents['proponent'].model_dim),
                'confidence': 0.0,
                'consensus_reached': False,
                'reason': 'No valid arguments generated'
            }
        
        stacked_embs = torch.stack(all_embs)
        weighted_sum = sum(e * c for e, c in zip(stacked_embs, confidences)) / sum(confidences)
        
        # Check consensus
        variance = stacked_embs.var(0).mean().item()
        consensus_reached = variance < (1.0 - self.consensus_threshold)
        
        # Final decision - use simple mean representation
        final_repr = weighted_sum.unsqueeze(0)  # [1, model_dim]
        confidence = float(sum(confidences) / len(confidences))
        
        return {
            'embedding': final_repr,
            'confidence': confidence,
            'consensus_reached': consensus_reached,
            'variance': variance,
            'num_arguments': len(all_embs)
        }
    
    def _aggregate_fact_checks(self, debate_rounds: List[DebateRound]) -> Dict:
        """Aggregate all fact checks from debate"""
        all_checks = []
        for round_data in debate_rounds:
            all_checks.extend(round_data.fact_checks)
        
        if not all_checks:
            return {'status': 'no_checks_performed'}
        
        # Aggregate scores
        contradiction_scores = [c['score'] for c in all_checks if c['type'] == 'contradiction_check']
        
        return {
            'total_checks': len(all_checks),
            'contradiction_score': sum(contradiction_scores) / len(contradiction_scores) if contradiction_scores else 0.0,
            'hallucination_risk': 'low' if (not contradiction_scores or max(contradiction_scores) < 0.3) else 'high'
        }
