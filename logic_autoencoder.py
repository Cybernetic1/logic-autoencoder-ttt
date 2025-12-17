#!/usr/bin/env python3
"""
Logic Auto-Encoder for Tic-Tac-Toe

Multi-layer architecture:
- Layer 1 (input): Learn ground predicates from board state
- Layer 2-3 (middle): Learn game patterns and rules
- Layer 4 (output): Predict next state

Focus: Observe what rules emerge, not optimal play.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class GroundPredicateLayer(nn.Module):
    """
    Layer 1: Learn basic predicates from board positions.
    Input: board state (9 positions x 3 states)
    Output: ground predicates for each position
    """
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        # Each position (x,y) gets encoded
        self.position_encoder = nn.Embedding(9, 16)  # 9 positions
        
        # Learn positional predicates: is_corner, is_center, is_edge
        self.is_corner = nn.Linear(16, 1)
        self.is_center = nn.Linear(16, 1)
        self.is_edge = nn.Linear(16, 1)
        
        # Learn state-dependent predicates: occupied_by_x, occupied_by_o, empty
        # Input: position encoding + one-hot state (3)
        self.state_predicates = nn.Linear(16 + 3, hidden_dim)
        
    def forward(self, board_state):
        """
        Args:
            board_state: (batch, 9, 3) - one-hot encoded board
        Returns:
            dict of predicates
        """
        batch_size = board_state.shape[0]
        positions = torch.arange(9, device=board_state.device)
        
        # Position encodings for all positions
        pos_enc = self.position_encoder(positions)  # (9, 16)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 9, 16)
        
        # Positional predicates (independent of state)
        is_corner = torch.sigmoid(self.is_corner(pos_enc))  # (batch, 9, 1)
        is_center = torch.sigmoid(self.is_center(pos_enc))  # (batch, 9, 1)
        is_edge = torch.sigmoid(self.is_edge(pos_enc))      # (batch, 9, 1)
        
        # State-dependent predicates
        pos_and_state = torch.cat([pos_enc, board_state], dim=-1)  # (batch, 9, 19)
        state_features = torch.tanh(self.state_predicates(pos_and_state))  # (batch, 9, hidden_dim)
        
        return {
            'is_corner': is_corner,
            'is_center': is_center,
            'is_edge': is_edge,
            'state_features': state_features,
            'position_encoding': pos_enc,
        }


class PatternRecognitionLayer(nn.Module):
    """
    Layer 2: Recognize game patterns (two-in-a-row, threats, etc.)
    Uses ground predicates from Layer 1.
    """
    
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        
        # Define winning lines
        self.lines = [
            [0, 1, 2],  # rows
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],  # cols
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],  # diagonals
            [2, 4, 6],
        ]
        
        # Learn to recognize patterns in lines
        # Input: features from 3 positions in a line
        self.line_pattern_detector = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )
        
        # Aggregate all line patterns
        self.pattern_aggregator = nn.Sequential(
            nn.Linear(32 * 8, hidden_dim),  # 8 lines
            nn.ReLU(),
        )
        
    def forward(self, ground_predicates, board_state):
        """
        Args:
            ground_predicates: dict from Layer 1
            board_state: (batch, 9, 3)
        Returns:
            pattern features
        """
        batch_size = board_state.shape[0]
        state_features = ground_predicates['state_features']  # (batch, 9, hidden_dim)
        
        # Process each line
        line_patterns = []
        for line in self.lines:
            # Get features for 3 positions in this line
            line_feats = state_features[:, line, :]  # (batch, 3, hidden_dim)
            line_feats_flat = line_feats.reshape(batch_size, -1)  # (batch, 3*hidden_dim)
            
            # Detect pattern in this line
            pattern = self.line_pattern_detector(line_feats_flat)  # (batch, 32)
            line_patterns.append(pattern)
        
        # Stack and aggregate all line patterns
        all_patterns = torch.stack(line_patterns, dim=1)  # (batch, 8, 32)
        all_patterns_flat = all_patterns.reshape(batch_size, -1)  # (batch, 8*32)
        
        aggregated = self.pattern_aggregator(all_patterns_flat)  # (batch, hidden_dim)
        
        return {
            'line_patterns': all_patterns,
            'aggregated_patterns': aggregated,
        }


class RuleLayer(nn.Module):
    """
    Layer 3: Learn game rules (decision logic)
    Combines patterns with positional info to make decisions.
    """
    
    def __init__(self, pattern_dim=64, hidden_dim=64):
        super().__init__()
        
        # Rule network: combine global patterns with local position info
        self.rule_network = nn.Sequential(
            nn.Linear(pattern_dim + 32 + 3, hidden_dim),  # pattern + pos_features + state
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
    def forward(self, ground_predicates, pattern_features, board_state):
        """
        Args:
            ground_predicates: dict from Layer 1
            pattern_features: dict from Layer 2
            board_state: (batch, 9, 3)
        Returns:
            rule activations for each position
        """
        batch_size = board_state.shape[0]
        aggregated_patterns = pattern_features['aggregated_patterns']  # (batch, pattern_dim)
        state_features = ground_predicates['state_features']  # (batch, 9, 32)
        
        # Broadcast pattern to all positions
        patterns_broadcast = aggregated_patterns.unsqueeze(1).expand(-1, 9, -1)  # (batch, 9, pattern_dim)
        
        # Combine global patterns with local info
        combined = torch.cat([
            patterns_broadcast,
            state_features,
            board_state,
        ], dim=-1)  # (batch, 9, pattern_dim + 32 + 3)
        
        # Apply rule network to each position
        rule_activations = self.rule_network(combined)  # (batch, 9, hidden_dim)
        
        return {
            'rule_activations': rule_activations,
        }


class OutputLayer(nn.Module):
    """
    Layer 4: Predict next state from rule activations.
    Output: probability distribution over next board states.
    """
    
    def __init__(self, rule_dim=64):
        super().__init__()
        
        # Predict next state for each position
        self.next_state_predictor = nn.Linear(rule_dim, 3)  # 3 states: empty, X, O
        
    def forward(self, rule_features):
        """
        Args:
            rule_features: dict from Layer 3
        Returns:
            next state predictions
        """
        rule_activations = rule_features['rule_activations']  # (batch, 9, rule_dim)
        
        # Predict next state for each position
        logits = self.next_state_predictor(rule_activations)  # (batch, 9, 3)
        
        return logits


class LogicAutoEncoder(nn.Module):
    """
    Complete multi-layer logic auto-encoder for TTT.
    """
    
    def __init__(self, hidden_dim=32, pattern_dim=64, rule_dim=64):
        super().__init__()
        
        self.layer1_ground = GroundPredicateLayer(hidden_dim=hidden_dim)
        self.layer2_patterns = PatternRecognitionLayer(input_dim=hidden_dim, hidden_dim=pattern_dim)
        self.layer3_rules = RuleLayer(pattern_dim=pattern_dim, hidden_dim=rule_dim)
        self.layer4_output = OutputLayer(rule_dim=rule_dim)
        
    def forward(self, board_state):
        """
        Args:
            board_state: (batch, 9, 3) - one-hot encoded
        Returns:
            next_state_logits: (batch, 9, 3)
            latent_features: dict of intermediate representations
        """
        # Layer 1: Ground predicates
        ground_preds = self.layer1_ground(board_state)
        
        # Layer 2: Pattern recognition
        patterns = self.layer2_patterns(ground_preds, board_state)
        
        # Layer 3: Rules
        rules = self.layer3_rules(ground_preds, patterns, board_state)
        
        # Layer 4: Output
        next_state_logits = self.layer4_output(rules)
        
        # Collect all latent features for analysis
        latent_features = {
            'ground_predicates': ground_preds,
            'patterns': patterns,
            'rules': rules,
        }
        
        return next_state_logits, latent_features
    
    def get_interpretable_features(self, board_state):
        """
        Extract interpretable features for analysis.
        """
        with torch.no_grad():
            _, latent = self.forward(board_state)
            
            return {
                'is_corner': latent['ground_predicates']['is_corner'].squeeze(-1),
                'is_center': latent['ground_predicates']['is_center'].squeeze(-1),
                'is_edge': latent['ground_predicates']['is_edge'].squeeze(-1),
                'line_patterns': latent['patterns']['line_patterns'],
                'rule_activations': latent['rules']['rule_activations'],
            }


def board_to_tensor(board: List[int]) -> torch.Tensor:
    """
    Convert board representation to one-hot tensor.
    
    Args:
        board: list of 9 values (0=empty, 1=X, -1=O)
    Returns:
        tensor of shape (9, 3) - one-hot encoded
    """
    # Map: 0 -> [1,0,0], 1 -> [0,1,0], -1 -> [0,0,1]
    board_tensor = torch.zeros(9, 3)
    for i, val in enumerate(board):
        if val == 0:
            board_tensor[i, 0] = 1
        elif val == 1:
            board_tensor[i, 1] = 1
        else:  # val == -1
            board_tensor[i, 2] = 1
    
    return board_tensor


def tensor_to_board(tensor: torch.Tensor) -> List[int]:
    """
    Convert one-hot tensor back to board representation.
    
    Args:
        tensor: (9, 3) or (9,) with class indices
    Returns:
        list of 9 values (0=empty, 1=X, -1=O)
    """
    if tensor.dim() == 2:
        indices = tensor.argmax(dim=-1)
    else:
        indices = tensor
    
    board = []
    for idx in indices:
        if idx == 0:
            board.append(0)
        elif idx == 1:
            board.append(1)
        else:
            board.append(-1)
    
    return board


def test_logic_autoencoder():
    """Test the logic autoencoder architecture."""
    print("Testing Logic Auto-Encoder Architecture")
    print("=" * 60)
    
    # Create model
    model = LogicAutoEncoder(hidden_dim=32, pattern_dim=64, rule_dim=64)
    
    # Test with a sample board
    # X in corners, O in center
    board = [1, 0, 1,
             0, -1, 0,
             1, 0, 0]
    
    board_tensor = board_to_tensor(board).unsqueeze(0)  # (1, 9, 3)
    print(f"Input board: {board}")
    print(f"Input shape: {board_tensor.shape}")
    
    # Forward pass
    logits, latent = model(board_tensor)
    print(f"\nOutput logits shape: {logits.shape}")
    
    # Get predictions
    predicted_state = logits.argmax(dim=-1)
    predicted_board = tensor_to_board(predicted_state[0])
    print(f"Predicted next board: {predicted_board}")
    
    # Examine learned predicates
    print("\n" + "=" * 60)
    print("Learned Ground Predicates (before training):")
    print("-" * 60)
    
    interpretable = model.get_interpretable_features(board_tensor)
    
    print("Is corner (should be high for positions 0,2,6,8):")
    print(interpretable['is_corner'][0].numpy())
    
    print("\nIs center (should be high for position 4):")
    print(interpretable['is_center'][0].numpy())
    
    print("\nIs edge (should be high for positions 1,3,5,7):")
    print(interpretable['is_edge'][0].numpy())
    
    print("\n" + "=" * 60)
    print("Architecture test complete!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_logic_autoencoder()
