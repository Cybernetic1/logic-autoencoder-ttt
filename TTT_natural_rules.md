# Natural-Form Logic Rules for TTT

## Philosophy

Instead of algorithmically optimal rules, design rules in forms that are **more likely to emerge from gradient descent** on random game data. These rules use coordinate-based symbolic representation without arithmetic.

## Core Representation

### Predicates
- `occupied(player, x, y)` - position (x,y) has player's piece
- `empty(x, y)` - position (x,y) is empty
- `same_x(x1, x2)` - both positions share x-coordinate
- `same_y(y1, y2)` - both positions share y-coordinate
- `on_main_diag(x, y)` - position on main diagonal {(1,1), (2,2), (3,3)}
- `on_anti_diag(x, y)` - position on anti-diagonal {(1,3), (2,2), (3,1)}
- `is_center(x, y)` - position is center {(2,2)}
- `is_corner(x, y)` - position is corner {(1,1), (1,3), (3,1), (3,3)}

### Symbols
- Players: `X`, `O`
- Coordinates: `1`, `2`, `3` (purely symbolic, no arithmetic)

---

## Natural Rule Forms

### Level 1: Direct Pattern Matching (Most Natural)

These emerge easily because they directly map observable patterns to actions.

**Rule 1.1: Three-in-a-row patterns (Row)**
```
∀ x, y1, y2, y3 where all_different(y1, y2, y3):
  IF occupied(me, x, y1) ∧ occupied(me, x, y2) ∧ empty(x, y3)
  THEN play(x, y3) with confidence=1.0
```

**Rule 1.2: Three-in-a-row patterns (Column)**
```
∀ x1, x2, x3, y where all_different(x1, x2, x3):
  IF occupied(me, x1, y) ∧ occupied(me, x2, y) ∧ empty(x3, y)
  THEN play(x3, y) with confidence=1.0
```

**Rule 1.3: Three-in-a-row (Main Diagonal)**
```
∀ x1, x2, x3 where all_different(x1, x2, x3):
  IF occupied(me, x1, x1) ∧ occupied(me, x2, x2) ∧ empty(x3, x3)
  THEN play(x3, x3) with confidence=1.0
```

**Rule 1.4: Three-in-a-row (Anti-Diagonal)**
```
Pattern matching for {(1,3), (2,2), (3,1)}:
  IF two occupied by me ∧ one empty
  THEN play empty position with confidence=1.0
```

**Rule 1.5: Block opponent (same patterns, replace 'me' with 'opponent')**
```
Same as rules 1.1-1.4 but with:
  occupied(opponent, ...) and confidence=0.9
```

### Level 2: Positional Heuristics (Very Natural)

These emerge from statistical regularities in game data.

**Rule 2.1: Center preference**
```
IF empty(2, 2)
THEN play(2, 2) with confidence=0.6
```

**Rule 2.2: Corner preference**
```
∀ x, y where is_corner(x, y):
  IF empty(x, y)
  THEN play(x, y) with confidence=0.4
```

**Rule 2.3: Edge as fallback**
```
∀ x, y where ¬is_corner(x, y) ∧ ¬is_center(x, y):
  IF empty(x, y)
  THEN play(x, y) with confidence=0.2
```

### Level 3: Relational Patterns (Moderately Natural)

Require learning relationships between positions, but no lookahead.

**Rule 3.1: Opposite corner response**
```
Diagonal pairs: {((1,1), (3,3)), ((1,3), (3,1))}
∀ (pos1, pos2) in diagonal_pairs:
  IF occupied(opponent, pos1) ∧ empty(pos2) ∧ is_early_game()
  THEN play(pos2) with confidence=0.5
```

**Rule 3.2: Form two-in-a-row**
```
∀ x, y1, y2 where y1 ≠ y2:
  IF occupied(me, x, y1) ∧ empty(x, y2) ∧ empty(x, third_y)
  THEN play(x, y2) with confidence=0.3
```

### Level 4: Fork Patterns (Less Natural - Require Lookahead)

These are harder to learn without explicit simulation/lookahead.

**Rule 4.1: Corner-center fork (X's opening)**
```
IF occupied(me, 1, 1) ∧ occupied(me, 2, 2) ∧ empty(3, 3)
THEN play(3, 3) with confidence=0.7
[This creates two winning lines: diagonal and corner]
```

**Rule 4.2: Edge-fork warning**
```
IF occupied(opponent, 1, 1) ∧ occupied(me, 2, 2) ∧ 
   occupied(opponent, edge_position)
THEN block_fork(...) 
[Complex - requires case analysis]
```

---

## Key Insights for Algorithm Design

### 1. **Grounded Predicates Should Be Learned**
- Don't hardcode `is_corner(x,y)` - let network learn which (x,y) pairs satisfy this
- Network output: probability that `is_corner` is true for each (x,y) combination
- Examples: `is_corner(1,1) = 1.0`, `is_corner(2,1) = 0.0`

### 2. **Rules as Soft Pattern Matchers**
Instead of exact boolean logic:
```python
# Soft version of Rule 1.1
score = σ(W1 @ [occupied(me,x,y1), occupied(me,x,y2), empty(x,y3), same_row])
play_probability(x,y3) ∝ score
```

### 3. **Rule Composition Hierarchy**
- **Layer 1**: Learn basic predicates (is_corner, same_x, etc.)
- **Layer 2**: Learn 2-in-a-row patterns using Layer 1
- **Layer 3**: Learn fork patterns using Layer 2
- Gradient flows through all layers

### 4. **Emergent Complexity**
Start training with:
- ✅ Level 1 rules (pattern matching)
- ✅ Level 2 rules (positional heuristics)
- ❓ Level 3 rules (may or may not emerge)
- ❌ Level 4 rules (unlikely without architecture changes)

### 5. **Data-Driven Discovery**
From random TTT games, networks will see:
- Many instances of 2-in-a-row → they learn to complete/block
- Center is often occupied early → they learn center preference  
- Forks are RARE in random play → they may not learn fork logic

---

## Proposed Algorithm Architecture

### Input Representation
```
For each position (x, y) ∈ {1,2,3} × {1,2,3}:
  state[x,y] = one_hot([empty, X, O])  # 3 values
Total input: 9 positions × 3 states = 27 dimensions
```

### Latent Logic State
```python
# Layer 1: Ground predicates (learned)
is_corner[x,y] = NN1([x, y])          # 9 values
is_center[x,y] = NN2([x, y])          # 9 values
same_x[x1,x2] = NN3([x1, x2])         # For relations
same_y[y1,y2] = NN4([y1, y2])

# Layer 2: Pattern detectors (learned from Layer 1 + input)
two_in_row[x,y,direction] = NN5([state, is_corner, same_x, ...])
has_threat[player,x,y] = NN6([...])

# Layer 3: Complex patterns (learned from Layer 2)
fork_opportunity[x,y] = NN7([two_in_row, has_threat, ...])
```

### Output
```
play_probability[x,y] = softmax(NN_final([all_latent_logic_state]))
```

### Loss Function
```
L = CrossEntropy(predicted_next_state, actual_next_state) 
    + α * Sparsity(latent_logic_state)  # Encourage crisp logic
    + β * Orthogonality(predicates)      # Encourage diverse concepts
```

---

## Advantages of This Approach

1. **Interpretable**: Each layer represents interpretable logic concepts
2. **Compositional**: Higher-level patterns built from lower-level ones
3. **Natural emergence**: Rules that appear in data will naturally be learned
4. **Graceful degradation**: Won't force learning of patterns that don't exist in data
5. **Symmetric**: Can use symmetric weight sharing for (x,y) coordinate predicates

## Next Steps

1. **Implement Transformer baseline** - see what patterns it learns implicitly
2. **Implement simple logic-AE** - start with Levels 1-2 only  
3. **Analyze learned predicates** - visualize what `is_corner`, `two_in_row` etc. learn
4. **Gradually add complexity** - add Level 3 if warranted by results
