"""
Compare Standard vs Slim Logic Networks on TTT Autoregressive Task

Tests both versions on the same task to compare:
1. Parameter counts
2. Training performance
3. Convergence speed
4. Final accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from neural_logic_core import LogicNetwork
from neural_logic_core_slim import SlimLogicNetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_ttt_data(num_samples=1000, num_props=9, prop_length=2):
    """
    Create synthetic TTT data for autoregressive prediction.
    
    Working memory format: [player, normalized_position]
    - player ∈ {-1, 0, 1} for {O, empty, X}
    - position ∈ [-1, +1]: (idx - 4) / 4
    
    Task: Predict next board state (9 positions × 3 states)
    """
    data = []
    
    for _ in range(num_samples):
        # Random number of moves (0-8)
        num_moves = np.random.randint(0, 8)
        
        # Create board
        board = [0] * 9
        available = list(range(9))
        
        for i in range(num_moves):
            if not available:
                break
            pos = np.random.choice(available)
            available.remove(pos)
            board[pos] = 1 if i % 2 == 0 else -1  # 1 for X, -1 for O
        
        current_player = 1 if num_moves % 2 == 0 else -1
        
        # Create next board (make one random move)
        if available:
            next_board = board.copy()
            next_pos = np.random.choice(available)
            next_board[next_pos] = current_player
        else:
            next_board = board
        
        # Convert to working memory: [player, position]
        current_wm = torch.zeros(9, 2)
        for i in range(9):
            current_wm[i, 0] = board[i]  # player value
            current_wm[i, 1] = (i - 4) / 4.0  # normalized position
        
        # Convert next state to one-hot for cross-entropy loss
        next_state_onehot = torch.zeros(9, 3)
        for i in range(9):
            if next_board[i] == 0:
                next_state_onehot[i, 0] = 1  # Empty
            elif next_board[i] == 1:
                next_state_onehot[i, 1] = 1  # X
            else:
                next_state_onehot[i, 2] = 1  # O
        
        data.append((current_wm, next_state_onehot))
    
    return data


def train_network(network, data, epochs=50, batch_size=32, lr=0.001, device='cpu'):
    """Train a logic network on TTT data."""
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        np.random.shuffle(data)
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            # Stack batch
            working_memory = torch.stack([b for b, _ in batch]).to(device)
            next_states = torch.stack([t for _, t in batch]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = network(working_memory)
            
            # Reshape predictions to match next_states: (batch, 9, 3)
            # Output is (batch, 27) → reshape to (batch, 9, 3)
            predictions = predictions.view(-1, 9, 3)
            
            loss = criterion(predictions, next_states)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy: percentage of correct cell predictions
            # For each position, check if predicted state matches
            pred_states = predictions.argmax(dim=2)  # (batch, 9)
            true_states = next_states.argmax(dim=2)  # (batch, 9)
            correct += (pred_states == true_states).sum().item()
            total += pred_states.numel()
        
        avg_loss = epoch_loss / (len(data) / batch_size)
        accuracy = correct / total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
    
    return losses, accuracies


def compare_networks():
    """Compare standard vs slim logic networks on TTT task."""
    # Configuration (matching successful training)
    L, W, output_dim = 2, 9, 27  # L=2 for [player,position], output=27 for 9x3
    M, J, I = 8, 2, 3  # 8 rules, 2 premises, 3 variable slots (matching train_logic_ae_unification.py)
    num_samples = 10000  # More data like successful training
    epochs = 100  # More epochs
    batch_size = 64
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Propositions: W={W}, length L={L}")
    print(f"  Rules: M={M}, Premises per rule: J={J}, Variable slots: I={I}")
    print(f"  Training: {num_samples} samples, {epochs} epochs")
    print(f"  Device: {device}")
    
    # Create data
    print(f"\nGenerating training data...")
    data = create_ttt_data(num_samples, W, L)
    print(f"  Generated {len(data)} samples")
    
    # Create networks
    print(f"\nCreating networks...")
    standard_net = LogicNetwork(L, W, output_dim, M, J, I)
    slim_net = SlimLogicNetwork(L, W, output_dim, M, J, I)
    
    standard_params = sum(p.numel() for p in standard_net.parameters())
    slim_params = sum(p.numel() for p in slim_net.parameters())
    
    print(f"  Standard network: {standard_params:,} parameters")
    print(f"  Slim network:     {slim_params:,} parameters")
    print(f"  Parameter reduction: {(1 - slim_params/standard_params)*100:.1f}%")
    
    # Train standard network
    print(f"\n{'='*80}")
    print("TRAINING STANDARD NETWORK")
    print('='*80)
    standard_losses, standard_accs = train_network(
        standard_net, data, epochs, batch_size, lr, device=device
    )
    
    # Train slim network
    print(f"\n{'='*80}")
    print("TRAINING SLIM NETWORK")
    print('='*80)
    slim_losses, slim_accs = train_network(
        slim_net, data, epochs, batch_size, lr, device=device
    )
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print('='*80)
    print(f"\nFinal Performance (last 10 epochs average):")
    print(f"  Standard - Loss: {np.mean(standard_losses[-10:]):.4f}, Accuracy: {np.mean(standard_accs[-10:]):.3f}")
    print(f"  Slim     - Loss: {np.mean(slim_losses[-10:]):.4f}, Accuracy: {np.mean(slim_accs[-10:]):.3f}")
    
    print(f"\nBest Performance:")
    print(f"  Standard - Best Accuracy: {max(standard_accs):.3f}")
    print(f"  Slim     - Best Accuracy: {max(slim_accs):.3f}")
    
    print(f"\nConvergence (epochs to reach 0.5 accuracy):")
    standard_conv = next((i for i, acc in enumerate(standard_accs) if acc >= 0.5), None)
    slim_conv = next((i for i, acc in enumerate(slim_accs) if acc >= 0.5), None)
    print(f"  Standard: {standard_conv if standard_conv else 'Did not converge'}")
    print(f"  Slim:     {slim_conv if slim_conv else 'Did not converge'}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(standard_losses, label='Standard', linewidth=2)
    ax1.plot(slim_losses, label='Slim', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(standard_accs, label='Standard', linewidth=2)
    ax2.plot(slim_accs, label='Slim', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('standard_vs_slim_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: standard_vs_slim_comparison.png")
    
    # Save networks
    torch.save(standard_net.state_dict(), 'standard_ttt_model.pt')
    torch.save(slim_net.state_dict(), 'slim_ttt_model.pt')
    print(f"Models saved to: standard_ttt_model.pt, slim_ttt_model.pt")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print('='*80)
    
    # Determine winner
    standard_score = np.mean(standard_accs[-10:])
    slim_score = np.mean(slim_accs[-10:])
    
    if abs(standard_score - slim_score) < 0.02:
        print("Both networks perform similarly!")
        print(f"Slim network uses {100*(1-slim_params/standard_params):.1f}% fewer parameters")
        print("→ Slim version recommended for parameter efficiency")
    elif standard_score > slim_score:
        print("Standard network performs better.")
        print(f"Accuracy difference: +{standard_score - slim_score:.3f}")
        print(f"But uses {standard_params - slim_params:,} more parameters")
        print("→ Trade-off: performance vs efficiency")
    else:
        print("Slim network performs better!")
        print(f"Accuracy difference: +{slim_score - standard_score:.3f}")
        print(f"AND uses {100*(1-slim_params/standard_params):.1f}% fewer parameters")
        print("→ Slim version is clearly superior")
    
    return {
        'standard': {'losses': standard_losses, 'accs': standard_accs, 'params': standard_params},
        'slim': {'losses': slim_losses, 'accs': slim_accs, 'params': slim_params}
    }


if __name__ == "__main__":
    results = compare_networks()
