#!/usr/bin/env python3
"""
Logic Auto-Encoder for TTT using Neural Logic Network with Fuzzy Unification.

This version uses the logic rules from neural_logic_core.py instead of 
traditional neural layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from neural_logic_core import LogicNetwork


def board_to_propositions(board):
    """
    Convert board to propositions (working memory).
    
    Args:
        board: list of 9 values (0=empty, 1=X, -1=O)
    
    Returns:
        propositions: (9, 2) tensor where each row is [player, position]
        - player ∈ {-1, 0, 1} for {O, empty, X}
        - position ∈ [-1, +1] normalized position: (idx - 4) / 4
    """
    props = torch.zeros(9, 2)
    for i, val in enumerate(board):
        props[i, 0] = val  # player
        props[i, 1] = (i - 4) / 4.0  # normalized position
    return props


def propositions_to_board(props):
    """
    Convert propositions back to board.
    
    Args:
        props: (9, 2) tensor or (9,) with player values
    
    Returns:
        list of 9 values (0=empty, 1=X, -1=O)
    """
    if props.dim() == 2:
        player_vals = props[:, 0]
    else:
        player_vals = props
    
    board = []
    for val in player_vals:
        if val < -0.3:
            board.append(-1)  # O
        elif val > 0.3:
            board.append(1)   # X
        else:
            board.append(0)   # empty
    
    return board


class LogicAutoEncoder(nn.Module):
    """
    Logic-based autoencoder using fuzzy unification rules.
    
    Task: Predict next board state from current state.
    """
    
    def __init__(self, num_rules=8, num_premises=2, var_slots=3):
        super().__init__()
        
        # Logic network: processes 9 propositions of length 2 [player, position]
        # Outputs 9 predictions (one per position) with 3 classes [empty, X, O]
        self.logic_net = LogicNetwork(
            prop_length=2,       # [player, position]
            num_props=9,         # 9 board positions
            output_dim=9 * 3,    # 9 positions × 3 states
            num_rules=num_rules,
            num_premises=num_premises,
            var_slots=var_slots,
        )
    
    def forward(self, board_state, return_details=False):
        """
        Args:
            board_state: (batch, 9, 3) - one-hot encoded board
            return_details: Whether to return rule details
        
        Returns:
            logits: (batch, 9, 3) - predictions for each position
            details: (optional) per-rule information
        """
        batch_size = board_state.shape[0]
        
        # Convert one-hot to propositions
        # Extract player values from one-hot: 0=empty, 1=X, -1=O
        working_memory = torch.zeros(batch_size, 9, 2, device=board_state.device)
        
        for b in range(batch_size):
            for i in range(9):
                # One-hot: [empty, X, O]
                if board_state[b, i, 0] > 0.5:
                    player = 0.0
                elif board_state[b, i, 1] > 0.5:
                    player = 1.0
                else:
                    player = -1.0
                
                working_memory[b, i, 0] = player
                working_memory[b, i, 1] = (i - 4) / 4.0  # normalized position
        
        # Process through logic network
        if return_details:
            output, details = self.logic_net(working_memory, return_details=True)
        else:
            output = self.logic_net(working_memory, return_details=False)
            details = None
        
        # Reshape output to (batch, 9, 3)
        logits = output.view(batch_size, 9, 3)
        
        return logits, details


class RandomTTTDataset(Dataset):
    """Generate random Tic-Tac-Toe games."""
    
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random game transition."""
        num_moves = np.random.randint(0, 8)
        
        board = [0] * 9
        available = list(range(9))
        
        # Random moves alternating X and O
        for i in range(num_moves):
            if not available:
                break
            pos = np.random.choice(available)
            available.remove(pos)
            board[pos] = 1 if i % 2 == 0 else -1
        
        # Current player
        current_player = 1 if num_moves % 2 == 0 else -1
        
        # Make one more random move
        if available:
            next_board = board.copy()
            next_pos = np.random.choice(available)
            next_board[next_pos] = current_player
        else:
            next_board = board
        
        # Convert to one-hot tensors
        current_state = self._to_onehot(board)
        next_state = self._to_onehot(next_board)
        
        return current_state, next_state
    
    def _to_onehot(self, board):
        """Convert board to one-hot encoding."""
        state = torch.zeros(9, 3)
        for i, val in enumerate(board):
            if val == 0:
                state[i, 0] = 1
            elif val == 1:
                state[i, 1] = 1
            else:
                state[i, 2] = 1
        return state


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (current_state, next_state) in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}", end='\r')
        
        current_state = current_state.to(device)
        next_state = next_state.to(device)
        
        # Forward pass
        logits, _ = model(current_state)
        
        # Loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 3),
            next_state.argmax(dim=-1).reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accuracy
        predictions = logits.argmax(dim=-1)
        targets = next_state.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for current_state, next_state in dataloader:
            current_state = current_state.to(device)
            next_state = next_state.to(device)
            
            logits, _ = model(current_state)
            
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 3),
                next_state.argmax(dim=-1).reshape(-1)
            )
            
            predictions = logits.argmax(dim=-1)
            targets = next_state.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader), correct / total


def analyze_learned_rules(model):
    """Analyze what logic rules the model learned."""
    print("\n" + "=" * 60)
    print("LEARNED LOGIC RULES")
    print("=" * 60)
    
    interpretation = model.logic_net.interpret_rules(
        prop_names=['player', 'position']
    )
    print(interpretation)


def test_specific_patterns(model, device):
    """Test if model learned specific game patterns."""
    print("\n" + "=" * 60)
    print("TESTING LEARNED PATTERNS")
    print("=" * 60)
    
    model.eval()
    
    test_cases = [
        {
            'name': 'Win completion (X X _)',
            'board': [1, 1, 0, 0, 0, 0, 0, 0, 0],
            'expected_pos': 2,
        },
        {
            'name': 'Block opponent (O O _)',
            'board': [0, 0, 0, -1, -1, 0, 0, 0, 0],
            'expected_pos': 5,
        },
        {
            'name': 'Diagonal win',
            'board': [1, 0, 0, 0, 1, 0, 0, 0, 0],
            'expected_pos': 8,
        },
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"Board: {test['board']}")
        
        # Convert to one-hot
        state = torch.zeros(1, 9, 3).to(device)
        for i, val in enumerate(test['board']):
            if val == 0:
                state[0, i, 0] = 1
            elif val == 1:
                state[0, i, 1] = 1
            else:
                state[0, i, 2] = 1
        
        with torch.no_grad():
            logits, details = model(state, return_details=True)
            probs = torch.softmax(logits[0], dim=-1)
            predicted = logits[0].argmax(dim=-1)
        
        # Check expected position
        expected_pos = test['expected_pos']
        prob_x = probs[expected_pos, 1].item()
        
        print(f"Expected move at position {expected_pos}: P(X)={prob_x:.3f}")
        
        if predicted[expected_pos].item() == 1:  # Predicted X
            print("✓ Correctly predicted!")
        else:
            print("✗ Did not predict correctly")
        
        # Show rule activations for this case
        if details:
            print(f"Rule confidence scores: ", end="")
            for i, rule_info in enumerate(details[:3]):
                conf = rule_info['confidence'][0].item()
                print(f"R{i+1}={conf:.2f} ", end="")
            print()


def main():
    """Main training loop."""
    print("Neural Logic Auto-Encoder for Tic-Tac-Toe")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with logic rules
    model = LogicAutoEncoder(num_rules=8, num_premises=2, var_slots=3)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    
    # Datasets
    train_dataset = RandomTTTDataset(num_samples=10000)
    val_dataset = RandomTTTDataset(num_samples=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 20
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Analysis
    analyze_learned_rules(model)
    test_specific_patterns(model, device)
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Prediction Accuracy')
    
    plt.tight_layout()
    plt.savefig('logic_ae_unification_training.png')
    print("\nTraining curves saved to logic_ae_unification_training.png")
    
    # Save model
    torch.save(model.state_dict(), 'logic_ae_unification_model.pt')
    print("Model saved to logic_ae_unification_model.pt")


if __name__ == "__main__":
    main()
