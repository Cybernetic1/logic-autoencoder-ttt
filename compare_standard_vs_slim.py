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
    
    Each proposition: [player, position]
    Task: Predict next board state
    """
    data = []
    
    for _ in range(num_samples):
        # Random board state
        board = torch.zeros(num_props, prop_length)
        
        # Random number of moves (1-9)
        num_moves = np.random.randint(1, 10)
        
        # Make random moves
        positions = np.random.permutation(num_props)[:num_moves]
        for i, pos in enumerate(positions):
            player = (i % 2) + 1  # Alternate between player 1 and 2
            board[pos, 0] = player  # Player ID
            board[pos, 1] = 1       # Occupied
        
        # Next move: random empty position
        empty_positions = [i for i in range(num_props) if board[i, 1] == 0]
        if empty_positions:
            next_pos = np.random.choice(empty_positions)
            next_player = (num_moves % 2) + 1
            
            # Target: one-hot over positions, weighted by player
            target = torch.zeros(num_props)
            target[next_pos] = 1.0
            
            data.append((board, target))
    
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
            boards = torch.stack([b for b, _ in batch]).to(device)
            targets = torch.stack([t for _, t in batch]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = network(boards)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accuracy: predicted position matches target
            pred_pos = outputs.argmax(dim=1)
            true_pos = targets.argmax(dim=1)
            correct += (pred_pos == true_pos).sum().item()
            total += len(batch)
        
        avg_loss = epoch_loss / (len(data) / batch_size)
        accuracy = correct / total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
    
    return losses, accuracies


def compare_networks():
    """Compare standard and slim networks on TTT task."""
    print("=" * 80)
    print("COMPARING STANDARD vs SLIM LOGIC NETWORKS ON TTT")
    print("=" * 80)
    
    # Configuration
    L, W, output_dim = 2, 9, 9
    M, J, I = 6, 3, 4
    num_samples = 2000
    epochs = 100
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
    print(f"  Reduction:        {100*(1-slim_params/standard_params):.1f}%")
    
    # Train standard network
    print(f"\n{'='*80}")
    print("TRAINING STANDARD NETWORK")
    print('='*80)
    standard_losses, standard_accs = train_network(
        standard_net, data, epochs, device=device
    )
    
    # Train slim network
    print(f"\n{'='*80}")
    print("TRAINING SLIM NETWORK")
    print('='*80)
    slim_losses, slim_accs = train_network(
        slim_net, data, epochs, device=device
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
