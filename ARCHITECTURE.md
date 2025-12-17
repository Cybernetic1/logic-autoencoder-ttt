# Architecture: Logic Auto-Encoder for TTT as AGI Simulation

## Core Concept

Use Tic-Tac-Toe gameplay (random moves) as a simulation environment for AGI research. The goal is to predict/model game dynamics using logic-based representations.

## Auto-Encoder Design

### Input/Output
- **Input**: Current game state
- **Output**: Predicted next game state (after random move)

### Latent Representation
- Form: **Logic propositions** (symbolic)
- Structure: Multiple layers of logic propositions
- The propositions across all layers = **complete state** of the auto-encoder

### Learning Components

1. **Fixed Structure**: 
   - Form of logic propositions is predefined
   
2. **Learnable Components**:
   - New predicates (relations/properties)
   - New constants (specific entities)
   - Logic rules (implemented as neural networks)

### Logic Rules as Neural Networks
- Each logic rule = a neural network
- NNs operate on logic propositions
- Can have multiple layers

## Future Integration

This latent logic state will serve as the **state representation** for subsequent reinforcement learning experiments.

## Key Advantage

Logic-based latent space should be more interpretable and compositional than standard neural latent vectors, potentially enabling better generalization and reasoning.

## Logic Representation for TTT

### Coordinate System
- Use (x,y) coordinates where x, y ∈ {1, 2, 3}
- Numbers are **pure symbols** (no arithmetic properties)
- Example propositions: occupied(X, 2, 3), player(X, x), winner(o), etc.

## Handling Stochasticity

### Challenge
Optimal TTT strategies can be **stochastic/probabilistic**. How can logic rules model this?

### Solutions

1. **Probabilistic Logic Networks**
   - Each logic rule outputs a probability distribution
   - Neural networks naturally output continuous values [0,1]
   - Can use softmax for multi-choice decisions
   
2. **Fuzzy/Soft Logic**
   - Truth values in [0,1] instead of {0,1}
   - Logic operations become differentiable (product t-norm, etc.)
   - Already compatible with neural network outputs

3. **Ensemble of Rules**
   - Multiple deterministic rules compete
   - Weighted combination gives stochastic behavior
   - Weights learned via NNs

4. **Latent Stochastic Variables**
   - Add random/noise predicates to logic state
   - Rules can condition on these
   - Enables stochastic decision-making

### Recommendation
Start with **fuzzy logic + softmax outputs** for action selection. This is most natural for neural network implementation and allows modeling probability distributions over next states.

## Baseline: Transformer Experiment

Before implementing logic-based autoencoder, establish a **Transformer baseline** for comparison.

### Purpose
- Measure how well standard deep learning (Transformer) can predict random TTT gameplay
- Provides performance benchmark for logic-based approach
- May reveal if interpretable logic structure offers advantages beyond raw prediction accuracy

### Experiment Design
- Input: Current board state (9 positions, 3 states each: empty/X/O)
- Output: Predicted next board state after random move
- Loss: Cross-entropy or MSE on predicted vs actual next state
- Training data: Games played with random moves

### Metrics to Track
1. Prediction accuracy (exact next state)
2. Move prediction accuracy (which cell was played)
3. Training efficiency (convergence speed, data requirements)
4. Model size

### Why This Helps
- If Transformer performs poorly → problem is inherently hard
- If Transformer performs well → logic approach needs to match/exceed this
- Establishes quantitative targets for logic-based model
