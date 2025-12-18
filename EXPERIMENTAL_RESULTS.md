# Experimental Results: Logic Auto-Encoder on Random TTT

## Experiment Setup
- **Architecture**: 4-layer logic autoencoder
  - Layer 1: Ground predicates (is_corner, is_center, is_edge)
  - Layer 2: Pattern recognition (line patterns)
  - Layer 3: Rule learning
  - Layer 4: Next state prediction
- **Training data**: 10,000 random TTT game transitions
- **Task**: Predict next board state after a random move

## Results

### Training Performance
- **Final accuracy**: ~89% on random game prediction
- **Convergence**: Stable after ~10 epochs
- **Loss**: CrossEntropy ~0.28

### What the Model Learned

✓ **Successfully learned**:
- The model achieves 89% accuracy at predicting random moves
- Learned to alternate players (always predicts opposite player)
- Some spatial awareness (different predictions for different positions)

✗ **Did NOT learn**:
- Ground predicates stayed at 0.5 (didn't specialize to corner/center/edge)
- Win completion patterns
- Blocking opponent threats
- Strategic center preference

## Key Discoveries

### Discovery 1: Random Play ≠ Optimal Play
The model learned to predict **random** moves, not **optimal** moves. This makes sense because:
- In random play, all empty positions have roughly equal probability
- Win opportunities appear rarely (and are often missed)
- Blocking threats appears rarely
- Strategic patterns (forks, opposite corners) virtually never appear

### Discovery 2: Ground Predicates Not Learned
The positional predicates (is_corner, is_center, is_edge) remained at 0.5 for all positions. This suggests:
- These predicates aren't useful for predicting random play
- The sparsity regularization wasn't strong enough
- The architecture may not provide enough gradient signal to these predicates

### Discovery 3: Player Alternation Learned Perfectly
When board has mostly X → model predicts O moves
When board has mostly O → model predicts X moves
This shows the model understands game structure at a basic level.

## Implications for Architecture

### Problem: Wrong Training Signal
Training on random games doesn't create pressure to learn strategic rules. The model optimizes for:
```
P(next_state | current_state, random_policy)
```

But we want it to learn:
```
P(next_state | current_state, optimal_policy)
```

### Possible Solutions

1. **Use optimal/semi-optimal game data**
   - Generate games using minimax or heuristic players
   - This will show win/block patterns more frequently
   - Fork patterns will appear in the data

2. **Add auxiliary tasks**
   - Predict winner from current state
   - Detect threats (two-in-a-row patterns)
   - Classify position types (corner/edge/center)
   - Force the model to learn useful predicates

3. **Curriculum learning**
   - Start with simple patterns (two-in-a-row)
   - Gradually increase game complexity
   - Add optimal moves progressively

4. **Reinforcement learning integration**
   - Use the logic layer as state representation
   - Train with RL to win games
   - This naturally creates pressure for strategic rules

## Next Steps

### Option A: Better Training Data
Modify dataset to include:
- 50% optimal/near-optimal games
- 50% random games
This will show the model what good play looks like while maintaining diversity.

### Option B: Multi-Task Learning
Add auxiliary prediction tasks:
- Is this a winning/losing position?
- How many two-in-a-row patterns exist?
- What type is each position? (corner/edge/center)

### Option C: Focus on Interpretability
Even with current results, we could:
- Visualize what the pattern layers are detecting
- Analyze the line_pattern activations
- See if ANY game-relevant structure emerged

## Conclusion

The experiment successfully demonstrated:
- ✓ The architecture works and trains
- ✓ Multi-layer logic representation is feasible
- ✓ Model can learn from game data

But revealed a fundamental issue:
- ✗ **Random game data insufficient for learning strategic rules**

**Recommendation**: Proceed with Option A (better training data) as it's closest to our goal of observing emergent strategic rules.
