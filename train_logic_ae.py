#!/usr/bin/env python3
"""
Training script for Logic Auto-Encoder on random TTT games.

Goal: Observe what rules emerge from random game data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from logic_autoencoder import LogicAutoEncoder, board_to_tensor, tensor_to_board


class RandomTTTDataset(Dataset):
    """Generate random Tic-Tac-Toe games on the fly."""
    
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random game transition."""
        # Start with empty board or partially filled
        num_moves = np.random.randint(0, 8)  # 0 to 7 existing moves
        
        board = [0] * 9
        available = list(range(9))
        
        # Random moves alternating X and O
        for i in range(num_moves):
            if not available:
                break
            pos = np.random.choice(available)
            available.remove(pos)
            board[pos] = 1 if i % 2 == 0 else -1
        
        # Determine whose turn it is
        current_player = 1 if num_moves % 2 == 0 else -1
        
        # Make one more random move for next state
        if available:
            next_board = board.copy()
            next_pos = np.random.choice(available)
            next_board[next_pos] = current_player
        else:
            next_board = board  # No move possible
        
        # Convert to tensors
        current_state = board_to_tensor(board)
        next_state = board_to_tensor(next_board)
        
        return current_state, next_state


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (current_state, next_state) in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}", end='\r')
        current_state = current_state.to(device)
        next_state = next_state.to(device)
        
        # Forward pass
        logits, latent = model(current_state)
        
        # Loss: cross-entropy for each position
        loss = F.cross_entropy(
            logits.reshape(-1, 3),
            next_state.argmax(dim=-1).reshape(-1)
        )
        
        # Add sparsity regularization on latent features (encourage crisp logic)
        sparsity_loss = 0
        ground_preds = latent['ground_predicates']
        for key in ['is_corner', 'is_center', 'is_edge']:
            pred = ground_preds[key]
            # Encourage values close to 0 or 1 (binary entropy)
            # -p*log(p) - (1-p)*log(1-p) is minimized when p=0 or p=1
            epsilon = 1e-7
            pred = torch.clamp(pred, epsilon, 1 - epsilon)
            sparsity = -torch.mean(pred * torch.log(pred) + (1 - pred) * torch.log(1 - pred))
            sparsity_loss += sparsity
        
        total_loss_val = loss + 0.1 * sparsity_loss  # Increased from 0.01
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()
        
        # Track accuracy
        predictions = logits.argmax(dim=-1)
        targets = next_state.argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        
        total_loss += loss.item()
        correct_predictions += correct
        total_predictions += targets.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for current_state, next_state in dataloader:
            current_state = current_state.to(device)
            next_state = next_state.to(device)
            
            logits, _ = model(current_state)
            
            loss = F.cross_entropy(
                logits.reshape(-1, 3),
                next_state.argmax(dim=-1).reshape(-1)
            )
            
            predictions = logits.argmax(dim=-1)
            targets = next_state.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            
            total_loss += loss.item()
            correct_predictions += correct
            total_predictions += targets.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def analyze_learned_predicates(model, device):
    """Analyze what the model has learned about positions."""
    print("\n" + "=" * 60)
    print("ANALYZING LEARNED GROUND PREDICATES")
    print("=" * 60)
    
    # Empty board
    empty_board = board_to_tensor([0] * 9).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.get_interpretable_features(empty_board)
    
    is_corner = features['is_corner'][0].cpu().numpy()
    is_center = features['is_center'][0].cpu().numpy()
    is_edge = features['is_edge'][0].cpu().numpy()
    
    print("\nPosition layout:")
    print("  0 1 2")
    print("  3 4 5")
    print("  6 7 8")
    
    print("\nIs Corner (expect high: 0,2,6,8):")
    print(f"  {is_corner[0]:.3f} {is_corner[1]:.3f} {is_corner[2]:.3f}")
    print(f"  {is_corner[3]:.3f} {is_corner[4]:.3f} {is_corner[5]:.3f}")
    print(f"  {is_corner[6]:.3f} {is_corner[7]:.3f} {is_corner[8]:.3f}")
    
    print("\nIs Center (expect high: 4):")
    print(f"  {is_center[0]:.3f} {is_center[1]:.3f} {is_center[2]:.3f}")
    print(f"  {is_center[3]:.3f} {is_center[4]:.3f} {is_center[5]:.3f}")
    print(f"  {is_center[6]:.3f} {is_center[7]:.3f} {is_center[8]:.3f}")
    
    print("\nIs Edge (expect high: 1,3,5,7):")
    print(f"  {is_edge[0]:.3f} {is_edge[1]:.3f} {is_edge[2]:.3f}")
    print(f"  {is_edge[3]:.3f} {is_edge[4]:.3f} {is_edge[5]:.3f}")
    print(f"  {is_edge[6]:.3f} {is_edge[7]:.3f} {is_edge[8]:.3f}")


def test_learned_rules(model, device):
    """Test if model learned basic game rules."""
    print("\n" + "=" * 60)
    print("TESTING LEARNED RULES")
    print("=" * 60)
    
    model.eval()
    
    # Test 1: Completing a win (X X _ -> X X X)
    print("\nTest 1: Win completion")
    board = [1, 1, 0,    # X X _
             0, -1, 0,   # _ O _
             0, -1, 0]   # _ O _
    print(f"Input:  {board}")
    
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(board_tensor)
        predicted = logits.argmax(dim=-1)[0]
    
    predicted_board = tensor_to_board(predicted.cpu())
    print(f"Output: {predicted_board}")
    print(f"Expected: X should play position 2")
    if predicted_board[2] == 1:
        print("✓ Correctly learned to complete win!")
    else:
        print("✗ Did not learn win completion")
    
    # Test 2: Blocking opponent
    print("\nTest 2: Block opponent's win")
    board = [1, 0, 0,    # X _ _
             0, -1, 0,   # _ O _
             0, -1, 0]   # _ O _
    print(f"Input:  {board}")
    
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(board_tensor)
        predicted = logits.argmax(dim=-1)[0]
    
    predicted_board = tensor_to_board(predicted.cpu())
    print(f"Output: {predicted_board}")
    print(f"Expected: X should block at position 1 (top middle)")
    if predicted_board[1] == 1:
        print("✓ Correctly learned to block!")
    else:
        print("✗ Did not learn blocking")
    
    # Test 3: Center preference on empty board
    print("\nTest 3: Center preference")
    board = [0] * 9
    print(f"Input:  {board}")
    
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(board_tensor)
        predicted = logits.argmax(dim=-1)[0]
    
    predicted_board = tensor_to_board(predicted.cpu())
    print(f"Output: {predicted_board}")
    print(f"Expected: Prefer center (position 4)")
    if predicted_board[4] == 1:
        print("✓ Learned center preference!")
    else:
        print("? Did not prefer center")


def main():
    """Main training loop."""
    print("Logic Auto-Encoder for Tic-Tac-Toe")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LogicAutoEncoder(hidden_dim=32, pattern_dim=64, rule_dim=64)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create datasets (smaller for faster testing)
    train_dataset = RandomTTTDataset(num_samples=10000)
    val_dataset = RandomTTTDataset(num_samples=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Analyze results
    analyze_learned_predicates(model, device)
    test_learned_rules(model, device)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Prediction Accuracy')
    
    plt.tight_layout()
    plt.savefig('logic_ae_training.png')
    print("\nTraining curves saved to logic_ae_training.png")
    
    # Save model
    torch.save(model.state_dict(), 'logic_ae_model.pt')
    print("Model saved to logic_ae_model.pt")


if __name__ == "__main__":
    main()
