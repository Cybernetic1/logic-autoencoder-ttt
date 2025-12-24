#!/usr/bin/env python3
"""
Tic-Tac-Toe Game Environment for RL Training
"""

import torch
import numpy as np
import random


class TicTacToeEnv:
    """Simple TTT environment for RL."""
    
    def __init__(self):
        self.reset()
        
        # Winning lines
        self.lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6],              # diagonals
        ]
    
    def reset(self):
        """Reset to empty board."""
        self.board = [0] * 9
        self.current_player = 1  # X starts
        self.done = False
        self.winner = None
        return self.board.copy()
    
    def get_valid_actions(self):
        """Return list of valid action indices."""
        return [i for i in range(9) if self.board[i] == 0]
    
    def check_winner(self):
        """Check if there's a winner. Returns 1, -1, or None."""
        for line in self.lines:
            vals = [self.board[i] for i in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return None
    
    def is_full(self):
        """Check if board is full."""
        return all(x != 0 for x in self.board)
    
    def step(self, action):
        """
        Take action.
        
        Args:
            action: integer 0-8
        
        Returns:
            next_state: board after action
            reward: reward for current player
            done: whether game is over
            info: dict with additional info
        """
        if self.done:
            raise ValueError("Game is already done. Call reset().")
        
        if action < 0 or action >= 9:
            raise ValueError(f"Invalid action: {action}")
        
        if self.board[action] != 0:
            # Invalid move - penalize and end
            return self.board.copy(), -10, True, {'invalid_move': True, 'winner': None}
        
        # Make move
        self.board[action] = self.current_player
        
        # Check for winner
        winner = self.check_winner()
        
        if winner is not None:
            self.done = True
            self.winner = winner
            # Reward from perspective of player who just moved
            reward = 1.0 if winner == self.current_player else -1.0
            return self.board.copy(), reward, True, {'winner': winner}
        
        # Check for draw
        if self.is_full():
            self.done = True
            return self.board.copy(), 0.0, True, {'winner': None, 'draw': True}
        
        # Game continues - switch player
        self.current_player = -self.current_player
        return self.board.copy(), 0.0, False, {}
    
    def get_board_tensor(self):
        """Get board as tensor for network input."""
        return torch.tensor(self.board, dtype=torch.float32)


class RandomOpponent:
    """Random player for training."""
    
    def choose_action(self, board):
        """Choose random valid action."""
        valid = [i for i in range(9) if board[i] == 0]
        return random.choice(valid) if valid else 0


class OptimalOpponent:
    """Near-optimal player using minimax."""
    
    def __init__(self):
        self.lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
    
    def choose_action(self, board):
        """Choose action using simple heuristics."""
        player = self._current_player(board)
        opponent = -player
        
        # 1. Win if possible
        move = self._find_winning_move(board, player)
        if move is not None:
            return move
        
        # 2. Block opponent
        move = self._find_winning_move(board, opponent)
        if move is not None:
            return move
        
        # 3. Take center
        if board[4] == 0:
            return 4
        
        # 4. Take corner
        corners = [0, 2, 6, 8]
        random.shuffle(corners)
        for c in corners:
            if board[c] == 0:
                return c
        
        # 5. Take any
        valid = [i for i in range(9) if board[i] == 0]
        return random.choice(valid) if valid else 0
    
    def _current_player(self, board):
        """Determine whose turn it is."""
        count = sum(1 for x in board if x != 0)
        return 1 if count % 2 == 0 else -1
    
    def _find_winning_move(self, board, player):
        """Find move that wins for player."""
        for line in self.lines:
            vals = [board[i] for i in line]
            if vals.count(player) == 2 and vals.count(0) == 1:
                for i in line:
                    if board[i] == 0:
                        return i
        return None


def play_episode(agent, opponent, env, epsilon=0.1, device='cpu'):
    """
    Play one episode.
    
    Args:
        agent: HierarchicalLogicNetwork
        opponent: opponent player
        env: TicTacToeEnv
        epsilon: exploration rate
        device: torch device
    
    Returns:
        trajectory: list of (state, action, reward, next_state, done)
        winner: 1, -1, or None
    """
    from hierarchical_logic_network import board_to_working_memory
    
    state = env.reset()
    trajectory = []
    
    while not env.done:
        # Agent's turn (X = 1)
        if env.current_player == 1:
            wm = board_to_working_memory(state, device)
            action = agent.choose_action(wm, epsilon=epsilon)
        else:
            # Opponent's turn (O = -1)
            action = opponent.choose_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        # Store transition from agent's perspective
        if env.current_player == -1 or done:  # After agent moved or game ended
            trajectory.append({
                'state': state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
            })
        
        state = next_state
    
    return trajectory, info.get('winner')


def test_environment():
    """Test the environment."""
    print("Testing Tic-Tac-Toe Environment")
    print("=" * 60)
    
    env = TicTacToeEnv()
    
    # Test basic gameplay
    print("Test 1: Basic game")
    state = env.reset()
    print(f"Initial state: {state}")
    
    # X plays center
    state, reward, done, info = env.step(4)
    print(f"After X plays 4: {state}, reward={reward}, done={done}")
    
    # O plays corner
    state, reward, done, info = env.step(0)
    print(f"After O plays 0: {state}, reward={reward}, done={done}")
    
    # Test winning
    print("\nTest 2: Winning scenario")
    env.reset()
    env.step(0)  # X
    env.step(3)  # O
    env.step(1)  # X
    env.step(4)  # O
    state, reward, done, info = env.step(2)  # X wins
    print(f"X wins: {state}")
    print(f"Reward: {reward}, Done: {done}, Winner: {info.get('winner')}")
    
    # Test opponents
    print("\nTest 3: Random opponent")
    env.reset()
    random_opp = RandomOpponent()
    
    for i in range(5):
        valid = env.get_valid_actions()
        if not valid:
            break
        action = random_opp.choose_action(env.board)
        print(f"Random chose: {action}")
        env.step(action)
        if env.done:
            break
    
    print("\nTest 4: Optimal opponent")
    env.reset()
    optimal_opp = OptimalOpponent()
    
    # Setup a scenario where O should block
    env.board = [1, 1, 0, 0, 0, 0, 0, 0, 0]  # X has two in a row
    env.current_player = -1
    
    action = optimal_opp.choose_action(env.board)
    print(f"Optimal opponent blocks at: {action} (should be 2)")
    
    print("\nEnvironment tests complete!")


if __name__ == "__main__":
    test_environment()
