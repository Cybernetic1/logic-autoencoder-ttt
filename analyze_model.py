#!/usr/bin/env python3
"""
Quick analysis of what the logic autoencoder learned.
"""

import torch
from logic_autoencoder import LogicAutoEncoder, board_to_tensor, tensor_to_board
import numpy as np


def analyze_model():
    """Load and analyze the trained model."""
    device = torch.device('cpu')
    model = LogicAutoEncoder(hidden_dim=32, pattern_dim=64, rule_dim=64)
    model.load_state_dict(torch.load('logic_ae_model.pt', map_location=device))
    model.eval()
    
    print("=" * 60)
    print("DETAILED ANALYSIS OF LEARNED MODEL")
    print("=" * 60)
    
    # Test various board positions to see what it predicts
    test_cases = [
        {
            'name': 'Win in row (X X _)',
            'board': [1, 1, 0, 0, 0, 0, 0, 0, 0],
            'expected_move': 2,
        },
        {
            'name': 'Win in column',
            'board': [1, 0, 0, 1, 0, 0, 0, 0, 0],
            'expected_move': 6,
        },
        {
            'name': 'Win on diagonal',
            'board': [1, 0, 0, 0, 1, 0, 0, 0, 0],
            'expected_move': 8,
        },
        {
            'name': 'Block opponent row',
            'board': [0, 0, 0, -1, -1, 0, 0, 0, 0],
            'expected_move': 5,
        },
        {
            'name': 'Block opponent diagonal',
            'board': [-1, 0, 0, 0, -1, 0, 0, 0, 0],
            'expected_move': 8,
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"Input board: {test['board']}")
        board_tensor = board_to_tensor(test['board']).unsqueeze(0)
        
        with torch.no_grad():
            logits, latent = model(board_tensor)
            predicted = logits.argmax(dim=-1)[0]
            probs = torch.softmax(logits[0], dim=-1)
        
        predicted_board = tensor_to_board(predicted)
        print(f"Output board: {predicted_board}")
        
        # Check if the expected move gets high probability for being played
        expected_pos = test['expected_move']
        prob_empty = probs[expected_pos, 0].item()
        prob_x = probs[expected_pos, 1].item()
        prob_o = probs[expected_pos, 2].item()
        
        print(f"Position {expected_pos} probabilities - Empty:{prob_empty:.3f}, X:{prob_x:.3f}, O:{prob_o:.3f}")
        
        if predicted_board[expected_pos] == 1:
            print("✓ Correctly predicted!")
        else:
            print("✗ Incorrect")
    
    # Check what the model does with different positions
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS: Two-in-a-row detection")
    print("=" * 60)
    
    # All possible two-in-a-row patterns for rows
    for row in range(3):
        print(f"\nRow {row}:")
        for config in [[1,1,0], [1,0,1], [0,1,1]]:
            board = [0] * 9
            for i, val in enumerate(config):
                board[row*3 + i] = val
            
            board_tensor = board_to_tensor(board).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(board_tensor)
                predicted = logits.argmax(dim=-1)[0]
            
            predicted_board = tensor_to_board(predicted)
            print(f"  {config} -> {predicted_board[row*3:(row+1)*3]} ", end="")
            
            # Check if it fills the empty spot
            empty_idx = config.index(0)
            if predicted_board[row*3 + empty_idx] == 1:
                print("✓")
            else:
                print("✗")


if __name__ == "__main__":
    analyze_model()
