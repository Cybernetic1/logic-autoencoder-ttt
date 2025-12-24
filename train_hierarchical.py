#!/usr/bin/env python3
"""
3-Phase Training Protocol for Hierarchical Logic Network

Phase 1: Concept Formation (AR only) - learn game dynamics
Phase 2: Concept Valuation (RL only) - learn strategy with frozen concepts
Phase 3: Joint Refinement (AR + RL) - co-adapt both objectives
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

from hierarchical_logic_network import (
    HierarchicalLogicNetwork, 
    board_to_working_memory,
    onehot_to_working_memory
)
from ttt_environment import TicTacToeEnv, RandomOpponent, OptimalOpponent, play_episode


class RandomTTTDataset(Dataset):
    """Generate random TTT game transitions for AR training."""
    
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        num_moves = np.random.randint(0, 8)
        board = [0] * 9
        available = list(range(9))
        
        for i in range(num_moves):
            if not available:
                break
            pos = np.random.choice(available)
            available.remove(pos)
            board[pos] = 1 if i % 2 == 0 else -1
        
        current_player = 1 if num_moves % 2 == 0 else -1
        
        if available:
            next_board = board.copy()
            next_pos = np.random.choice(available)
            next_board[next_pos] = current_player
        else:
            next_board = board
        
        current_state = self._to_onehot(board)
        next_state = self._to_onehot(next_board)
        
        return current_state, next_state
    
    def _to_onehot(self, board):
        state = torch.zeros(9, 3)
        for i, val in enumerate(board):
            if val == 0:
                state[i, 0] = 1
            elif val == 1:
                state[i, 1] = 1
            else:
                state[i, 2] = 1
        return state


class ReplayBuffer:
    """Experience replay buffer for RL."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


def train_phase1_ar(model, train_loader, optimizer, device, num_epochs=10):
    """
    Phase 1: Train autoregressive prediction (concept formation).
    Logic rules learn to capture game dynamics.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: CONCEPT FORMATION (Autoregressive Training)")
    print("=" * 60)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (current_state, next_state) in enumerate(train_loader):
            current_state = current_state.to(device)
            next_state = next_state.to(device)
            
            # Convert to working memory
            wm = onehot_to_working_memory(current_state, device)
            
            # Forward pass (AR mode)
            next_pred, _ = model.forward_ar(wm)
            
            # Loss
            loss = nn.functional.cross_entropy(
                next_pred.reshape(-1, 3),
                next_state.argmax(dim=-1).reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_ar_parameters(), 1.0)
            optimizer.step()
            
            # Stats
            predictions = next_pred.argmax(dim=-1)
            targets = next_state.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    return losses


def train_phase2_rl(model, env, replay_buffer, optimizer, device, 
                    num_episodes=1000, batch_size=32, gamma=0.99):
    """
    Phase 2: Train RL value function (concept valuation).
    Logic rules are FROZEN. Value network learns which concepts matter for winning.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: CONCEPT VALUATION (RL Training with Frozen Concepts)")
    print("=" * 60)
    
    model.freeze_logic_rules()
    model.train()
    
    opponent = RandomOpponent()
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    episode_rewards = []
    losses = []
    
    for episode in range(num_episodes):
        # Play episode
        trajectory, winner = play_episode(model, opponent, env, epsilon, device)
        
        # Store transitions
        episode_reward = 0
        for trans in trajectory:
            replay_buffer.push(
                trans['state'],
                trans['action'],
                trans['reward'],
                trans['next_state'],
                trans['done']
            )
            episode_reward += trans['reward']
        
        episode_rewards.append(episode_reward)
        
        # Train if enough data
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            # Convert to working memory
            wm = torch.stack([board_to_working_memory(s.tolist(), device).squeeze(0) 
                             for s in states])
            next_wm = torch.stack([board_to_working_memory(s.tolist(), device).squeeze(0)
                                  for s in next_states])
            
            # Q-learning update
            q_values, _ = model.forward_rl(wm)
            next_q_values, _ = model.forward_rl(next_wm)
            
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + gamma * next_q_value * (1 - dones)
            
            loss = nn.functional.mse_loss(q_value, expected_q_value.detach())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_rl_parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode+1}/{num_episodes} - "
                  f"Avg Reward: {avg_reward:.3f}, Loss: {avg_loss:.4f}, ε: {epsilon:.3f}")
    
    return episode_rewards, losses


def train_phase3_joint(model, train_loader, env, replay_buffer, 
                       ar_optimizer, rl_optimizer, device,
                       num_iterations=500, lambda_ar=0.5, lambda_rl=0.5):
    """
    Phase 3: Joint refinement (co-training).
    Both AR and RL objectives active. Logic rules refine to serve both purposes.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: JOINT REFINEMENT (AR + RL Co-training)")
    print("=" * 60)
    
    model.unfreeze_logic_rules()
    model.train()
    
    opponent = OptimalOpponent()  # Harder opponent for phase 3
    epsilon = 0.2
    gamma = 0.99
    batch_size = 32
    
    ar_losses = []
    rl_losses = []
    episode_rewards = []
    
    train_iter = iter(train_loader)
    
    for iteration in range(num_iterations):
        # === AR UPDATE ===
        try:
            current_state, next_state = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            current_state, next_state = next(train_iter)
        
        current_state = current_state.to(device)
        next_state = next_state.to(device)
        wm = onehot_to_working_memory(current_state, device)
        
        next_pred, concepts = model.forward_ar(wm)
        ar_loss = nn.functional.cross_entropy(
            next_pred.reshape(-1, 3),
            next_state.argmax(dim=-1).reshape(-1)
        )
        
        # === RL UPDATE ===
        # Play episode to collect data
        if iteration % 5 == 0:  # Play every 5 iterations
            trajectory, winner = play_episode(model, opponent, env, epsilon, device)
            ep_reward = sum(t['reward'] for t in trajectory)
            episode_rewards.append(ep_reward)
            
            for trans in trajectory:
                replay_buffer.push(
                    trans['state'], trans['action'], trans['reward'],
                    trans['next_state'], trans['done']
                )
        
        # RL training step
        rl_loss = torch.tensor(0.0, device=device)
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            
            wm_rl = torch.stack([board_to_working_memory(s.tolist(), device).squeeze(0) 
                                for s in states])
            next_wm_rl = torch.stack([board_to_working_memory(s.tolist(), device).squeeze(0)
                                     for s in next_states])
            
            q_values, _ = model.forward_rl(wm_rl)
            next_q_values, _ = model.forward_rl(next_wm_rl)
            
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + gamma * next_q_value * (1 - dones)
            
            rl_loss = nn.functional.mse_loss(q_value, expected_q_value.detach())
        
        # === JOINT UPDATE ===
        total_loss = lambda_ar * ar_loss + lambda_rl * rl_loss
        
        ar_optimizer.zero_grad()
        rl_optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ar_optimizer.step()
        rl_optimizer.step()
        
        ar_losses.append(ar_loss.item())
        rl_losses.append(rl_loss.item() if isinstance(rl_loss, torch.Tensor) else 0)
        
        if (iteration + 1) % 50 == 0:
            avg_ar_loss = np.mean(ar_losses[-50:])
            avg_rl_loss = np.mean(rl_losses[-50:])
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            
            print(f"Iter {iteration+1}/{num_iterations} - "
                  f"AR Loss: {avg_ar_loss:.4f}, RL Loss: {avg_rl_loss:.4f}, "
                  f"Reward: {avg_reward:.3f}")
    
    return ar_losses, rl_losses, episode_rewards


def main():
    """Execute 3-phase training protocol."""
    print("Hierarchical Logic Network: 3-Phase Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = HierarchicalLogicNetwork(
        num_rules=6,
        num_premises=2,
        var_slots=3,
        value_hidden_dim=32
    )
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create datasets
    train_dataset = RandomTTTDataset(num_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    env = TicTacToeEnv()
    replay_buffer = ReplayBuffer(capacity=5000)
    
    # === PHASE 1: AR Training ===
    ar_optimizer = optim.Adam(model.get_ar_parameters(), lr=0.001)
    phase1_losses = train_phase1_ar(model, train_loader, ar_optimizer, device, num_epochs=15)
    
    # === PHASE 2: RL Training ===
    rl_optimizer = optim.Adam(model.get_rl_parameters(), lr=0.001)
    phase2_rewards, phase2_losses = train_phase2_rl(
        model, env, replay_buffer, rl_optimizer, device, num_episodes=500
    )
    
    # === PHASE 3: Joint Training ===
    ar_optimizer = optim.Adam(model.get_ar_parameters(), lr=0.0005)
    rl_optimizer = optim.Adam(model.get_rl_parameters(), lr=0.0005)
    phase3_ar_losses, phase3_rl_losses, phase3_rewards = train_phase3_joint(
        model, train_loader, env, replay_buffer,
        ar_optimizer, rl_optimizer, device,
        num_iterations=300, lambda_ar=0.4, lambda_rl=0.6
    )
    
    # === ANALYSIS ===
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - ANALYZING LEARNED RULES")
    print("=" * 60)
    
    interpretation = model.logic_rules.interpret_rules(prop_names=['player', 'position'])
    print(interpretation)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(phase1_losses)
    axes[0, 0].set_title('Phase 1: AR Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(phase2_rewards)
    axes[0, 1].set_title('Phase 2: RL Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    
    axes[1, 0].plot(phase3_ar_losses, label='AR', alpha=0.7)
    axes[1, 0].plot(phase3_rl_losses, label='RL', alpha=0.7)
    axes[1, 0].set_title('Phase 3: Joint Training Losses')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    axes[1, 1].plot(phase3_rewards)
    axes[1, 1].set_title('Phase 3: RL Rewards')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('hierarchical_training_results.png')
    print("\nPlots saved to hierarchical_training_results.png")
    
    # Save model
    torch.save(model.state_dict(), 'hierarchical_logic_model.pt')
    print("Model saved to hierarchical_logic_model.pt")


if __name__ == "__main__":
    main()
