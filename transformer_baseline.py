#!/usr/bin/env python3
"""
Transformer baseline for TTT next-state prediction.
Trains on random gameplay to predict next board state.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add gym-tictactoe to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-tictactoe'))

import gym
import gym_tictactoe


class TTTDataset(Dataset):
    """Dataset of (current_state, next_state) pairs from random gameplay."""
    
    def __init__(self, num_games=10000):
        self.data = []
        self.generate_data(num_games)
    
    def generate_data(self, num_games):
        """Generate training data from random TTT games."""
        env = gym.make('TicTacToe-v0')
        
        print(f"Generating {num_games} random games...")
        for game_idx in range(num_games):
            if game_idx % 1000 == 0:
                print(f"  Game {game_idx}/{num_games}")
            
            state = env.reset()
            done = False
            
            while not done:
                # Get valid actions
                valid_actions = [a for a in range(9) if env.is_valid(a)]
                if not valid_actions:
                    break
                
                # Random action
                action = np.random.choice(valid_actions)
                
                # Store (current_state, action, next_state)
                current_state = state.copy()
                next_state, reward, done, info = env.step(action)
                
                self.data.append({
                    'current': self.state_to_tensor(current_state),
                    'next': self.state_to_tensor(next_state),
                    'action': action
                })
                
                state = next_state
        
        print(f"Generated {len(self.data)} state transitions")
    
    def state_to_tensor(self, state):
        """Convert board state to tensor.
        State is 3x3 board with: 0=empty, 1=X, 2=O
        Convert to one-hot: shape (9, 3)
        """
        state_flat = state.flatten()  # (9,)
        one_hot = np.zeros((9, 3), dtype=np.float32)
        for i, val in enumerate(state_flat):
            one_hot[i, int(val)] = 1.0
        return torch.from_numpy(one_hot)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class TTTTransformer(nn.Module):
    """Simple Transformer for TTT state prediction."""
    
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        
        # Input: 9 positions × 3 states (one-hot) = 9×3
        # Treat each position as a token with 3-dim embedding
        self.input_proj = nn.Linear(3, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: predict next state for each position
        self.output_proj = nn.Linear(d_model, 3)
    
    def forward(self, x):
        """
        x: (batch, 9, 3) - current board state (one-hot)
        returns: (batch, 9, 3) - predicted next state (logits)
        """
        # Project input
        x = self.input_proj(x)  # (batch, 9, d_model)
        
        # Transformer
        x = self.transformer(x)  # (batch, 9, d_model)
        
        # Output projection
        x = self.output_proj(x)  # (batch, 9, 3)
        
        return x


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        current = batch['current'].to(device)  # (batch, 9, 3)
        next_state = batch['next'].to(device)  # (batch, 9, 3)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(current)  # (batch, 9, 3)
        
        # Loss: cross-entropy per position
        # Reshape for cross_entropy: (batch*9, 3) and (batch*9)
        pred_flat = pred.reshape(-1, 3)
        target_flat = next_state.argmax(dim=2).reshape(-1)
        
        loss = criterion(pred_flat, target_flat)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        pred_classes = pred.argmax(dim=2)  # (batch, 9)
        target_classes = next_state.argmax(dim=2)  # (batch, 9)
        correct += (pred_classes == target_classes).sum().item()
        total += pred_classes.numel()
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    exact_match = 0
    move_correct = 0
    move_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            current = batch['current'].to(device)
            next_state = batch['next'].to(device)
            action = batch['action']
            
            pred = model(current)
            
            # Loss
            pred_flat = pred.reshape(-1, 3)
            target_flat = next_state.argmax(dim=2).reshape(-1)
            loss = criterion(pred_flat, target_flat)
            total_loss += loss.item()
            
            # Per-position accuracy
            pred_classes = pred.argmax(dim=2)
            target_classes = next_state.argmax(dim=2)
            correct += (pred_classes == target_classes).sum().item()
            total += pred_classes.numel()
            
            # Exact board match
            exact_match += (pred_classes == target_classes).all(dim=1).sum().item()
            
            # Move prediction accuracy (which cell changed)
            # Find cells that changed from current to next
            current_classes = current.argmax(dim=2)
            changed_mask = (current_classes != target_classes)
            pred_changed = (current_classes != pred_classes)
            
            # Check if predicted change matches actual change
            for i in range(len(action)):
                if changed_mask[i].any():
                    actual_move = changed_mask[i].nonzero(as_tuple=True)[0][0].item()
                    if pred_changed[i].any():
                        pred_move = pred_changed[i].nonzero(as_tuple=True)[0]
                        if len(pred_move) > 0:
                            pred_move = pred_move[0].item()
                            if pred_move == actual_move:
                                move_correct += 1
                    move_total += 1
    
    return {
        'loss': total_loss / len(dataloader),
        'position_acc': correct / total,
        'exact_match': exact_match / len(dataloader.dataset),
        'move_acc': move_correct / move_total if move_total > 0 else 0
    }


def main():
    # Hyperparameters
    NUM_GAMES = 20000
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("\n=== Generating Training Data ===")
    dataset = TTTDataset(num_games=NUM_GAMES)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Model
    print("\n=== Creating Model ===")
    model = TTTTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("\n=== Training ===")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Position Acc: {val_metrics['position_acc']:.4f}")
        print(f"  Val Exact Match: {val_metrics['exact_match']:.4f}, Val Move Acc: {val_metrics['move_acc']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'transformer_best.pth')
            print(f"  ✓ Saved best model")
    
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
