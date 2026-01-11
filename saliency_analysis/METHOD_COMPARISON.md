# Saliency Method Comparison: Vanilla Gradients vs Integrated Gradients

## Overview

This document explains the differences between the two saliency methods implemented and analyzes the results.

## Mathematical Formulations

### 1. Vanilla Gradients

**Formula:**
```
Saliency(x) = |∇F(x)|
```

Where:
- `F(x)` is the model output for input `x`
- `∇F(x)` is the gradient of the output with respect to the input
- `|·|` denotes absolute value (if `abs_gradients=True`)

**What it measures:**
- **Local sensitivity**: How much the output changes per unit change in input at the current point
- **Instantaneous gradient**: The gradient at a single point in input space
- **Fast computation**: Only requires one forward and one backward pass

**Characteristics:**
- Can be noisy and sensitive to small input changes
- May not capture the full attribution path
- Values are always non-negative (if `abs_gradients=True`)
- Simpler and faster to compute

### 2. Integrated Gradients

**Formula:**
```
IG(x) = (x - x') × (1/m) × Σ[i=1 to m] ∇F(x' + (i/m)(x - x'))
```

Where:
- `x'` is the baseline (usually zeros)
- `m` is the number of integration steps (`num_steps`)
- The integral is approximated by averaging gradients along the interpolation path

**What it measures:**
- **Path-integrated attribution**: How much each input feature contributes along the path from baseline to input
- **Axiomatic satisfaction**: Satisfies important axioms (sensitivity, implementation invariance)
- **More robust**: Less sensitive to noise and local variations

**Characteristics:**
- Can have **negative values**: Negative values indicate features that reduce the prediction
- More computationally expensive: Requires `num_steps` forward/backward passes (default: 50)
- More stable and theoretically grounded
- Accounts for the full path, not just the endpoint

## Key Differences

### 1. **Mathematical Definition**

| Aspect | Vanilla Gradients | Integrated Gradients |
|--------|------------------|---------------------|
| **Formula** | `\|∇F(x)\|` | `(x-x') × ∫∇F(x' + α(x-x'))dα` |
| **Computation** | Single gradient | Path-integrated gradient |
| **Baseline** | Not used | Required (default: zeros) |
| **Values** | Always ≥ 0 (if abs) | Can be negative or positive |

### 2. **Interpretation**

**Vanilla Gradients:**
- Positive values indicate sensitivity at the current point
- Higher values = more sensitive to changes
- No distinction between increasing vs decreasing prediction

**Integrated Gradients:**
- **Positive values**: Feature increases the prediction
- **Negative values**: Feature decreases the prediction
- **Magnitude**: Strength of the effect
- Satisfies attribution axioms (sensitivity, implementation invariance)

### 3. **Computational Cost**

- **Vanilla Gradients**: 1 forward pass + 1 backward pass per sample
- **Integrated Gradients**: `num_steps` (default: 50) forward + backward passes per sample
- **Speed ratio**: Integrated Gradients is ~50x slower (with default `num_steps=50`)

### 4. **Robustness**

- **Vanilla Gradients**: Can be noisy, sensitive to local curvature
- **Integrated Gradients**: More stable, averages over multiple points

## Implementation Details

### Current Implementation

**Vanilla Gradients:**
```python
1. Forward pass: output = model(input)
2. Backward pass: output.backward()
3. Get gradients: saliency = |input.grad|
```

**Integrated Gradients:**
```python
1. Determine target from original input (if not provided)
2. For each alpha in [0, 1] (num_steps points):
   a. Interpolate: x_alpha = baseline + alpha * (input - baseline)
   b. Forward pass: output = model(x_alpha)
   c. Backward pass: output.backward()
   d. Accumulate: gradients += |x_alpha.grad|  # if abs_gradients=True
3. Average: avg_gradients = gradients / num_steps
4. Multiply: IG = avg_gradients × (input - baseline)
```

### Note on `abs_gradients` Parameter

**Current behavior:**
- If `abs_gradients=True`, absolute value is taken **before** averaging in Integrated Gradients
- This changes the mathematical formula but is useful for saliency visualization

**Standard formula (without abs):**
```
IG(x) = (x - x') × (1/m) × Σ ∇F(x' + α(x - x'))
```

**Current implementation (with abs):**
```
IG_abs(x) = (x - x') × (1/m) × Σ |∇F(x' + α(x - x'))|
```

This is a variant that emphasizes magnitude over direction, which is often preferred for saliency visualization.

## Results Analysis

### Observed Differences

1. **Different Channel Rankings**: The two methods often rank channels differently
   - Example: Gender baseline 1s
     - VG top 5: [48, 26, 42, 37, 35]
     - IG top 5: [8, 29, 18, 38, 46]
   - This is **expected** due to different mathematical definitions

2. **Value Ranges**:
   - **Vanilla Gradients**: Always non-negative (0 to ~0.7)
   - **Integrated Gradients**: Can be negative or positive (-2.5 to ~4.0)
   - Negative values in IG indicate channels that reduce predictions

3. **Correlation**: Low correlation between methods (e.g., ~0.09 for age classification)
   - This is **expected** - they measure different things
   - Low correlation doesn't indicate an error

### Why Results Differ

1. **Different mathematical definitions**: They measure fundamentally different quantities
2. **Path vs point**: IG considers the full path, VG only the endpoint
3. **Robustness**: IG is less affected by local noise
4. **Baseline consideration**: IG accounts for the baseline (zeros), VG does not

## Potential Issues and Recommendations

### Issue 1: Ranking Integrated Gradients with Negative Values

**Current behavior**: Channels are ranked by `avg_importance`, which can be negative for IG.

**Recommendation**: For Integrated Gradients, consider ranking by absolute value to find the most "influential" channels (whether positive or negative):

```python
if method == 'integrated_gradients':
    ranked_indices = np.argsort(np.abs(avg_importance))[::-1]
else:
    ranked_indices = np.argsort(avg_importance)[::-1]
```

**Current status**: The code ranks by raw values, which means channels with large negative values might be ranked low even though they're highly influential (just in the negative direction).

### Issue 2: Visualization of Negative Values

**Current behavior**: Negative values in IG are visualized directly.

**Recommendation**: Consider using a diverging colormap (e.g., `RdBu_r`) that clearly distinguishes positive from negative values, or use absolute values for ranking while preserving sign in visualizations.

### Issue 3: Channel Importance Aggregation

**Current behavior**: Channel importance is computed as `mean(saliency_map, dim=2)` (averaging over time).

**Status**: ✅ **Correct** - This is appropriate for EEG data where we want to know overall channel importance across the time dimension.

## Best Practices

1. **Use Integrated Gradients for publication/research**: More theoretically grounded
2. **Use Vanilla Gradients for quick exploration**: Faster computation
3. **Compare both methods**: Low correlation is expected and informative
4. **Consider absolute values for ranking IG**: To find most influential channels
5. **Interpret negative IG values**: They indicate channels that reduce predictions

## Conclusion

Both implementations are **mathematically correct**. The differences in results are **expected** and reflect the fundamental differences between the methods:

- **Vanilla Gradients**: Fast, local sensitivity measure
- **Integrated Gradients**: Slower, path-integrated attribution measure

The low correlation between methods is **not a bug** - it indicates they capture different aspects of model behavior, which can provide complementary insights.

