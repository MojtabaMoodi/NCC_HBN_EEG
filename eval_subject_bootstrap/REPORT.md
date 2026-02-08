# Subject-Level Bootstrap Evaluation Report
## EEG Age-Group Classification

*Example report from a single evaluation run. For how to run the evaluation, see [README.md](README.md).*

**Date:** January 16, 2026  
**Evaluation Method:** Subject-level bootstrap with stratified random baseline comparison

---

## Methodology

### Subject-Level Evaluation
This evaluation aggregates predictions from multiple EEG segments per participant to a single prediction per participant. This approach is essential because:
- Segments from the same participant are not independent
- The goal is to predict participant characteristics, not segment characteristics
- Ensures proper statistical inference (subjects are independent units)

**Aggregation Method:** Mean probability voting
- For each participant, compute mean probability vector across all their segments
- Final prediction: `argmax(mean_probabilities)`

### Stratified Random Baseline
A subject-level stratified random baseline was generated using training set class proportions:
- Class 0 (<8.5 years): 35.8%
- Class 1 (8.5-12.5 years): 41.1%
- Class 2 (>12.5 years): 23.1%

This baseline reflects the true class distribution in the training population and provides a fair comparison at the subject level.

### Bootstrap Uncertainty Estimation
- **Method:** Subject-level (cluster) bootstrap with 2,000 replicates
- **CI Calculation:** 95% confidence intervals via percentile method [2.5%, 97.5%]
- **Delta Analysis:** Bootstrap CI for (model metric − random baseline metric)
- **Statistical Test:** Permutation test (1,000 replicates) for primary metric

---

## Dataset

- **Training Subjects:** 2,147
  - Class 0: 768 (35.8%)
  - Class 1: 883 (41.1%)
  - Class 2: 496 (23.1%)

- **Test Subjects:** 461
- **Test Segments:** 66,286
- **Segments per Subject:** 
  - Min: 5
  - Median: 163
  - Max: 207

- **Segment Length:** 4 seconds
- **Task Types:** Both active and passive tasks included

---

## Models Evaluated

1. **CNN** - Age classification CNN model
2. **ResNet-34** - ResNet-34 architecture for age classification
3. **LaBraM** - LaBraM base model (patch size 200) finetuned for age classification

All models were trained on 4-second EEG segments and evaluated using their best checkpoints.

---

## Results

### Primary Metric: Balanced Accuracy

| Model | Balanced Accuracy | 95% CI | Delta vs Random | Delta 95% CI | P(Δ > 0) |
|-------|-------------------|--------|-----------------|--------------|----------|
| **ResNet** | **0.6545** | [0.6109, 0.6995] | **0.3712** | [0.2591, 0.3859] | 1.0000 |
| LaBraM | 0.6230 | [0.5809, 0.6649] | 0.3397 | [0.2290, 0.3501] | 1.0000 |
| CNN | 0.6242 | [0.5786, 0.6688] | 0.3140 | [0.2269, 0.3528] | 1.0000 |
| Random Baseline | 0.2834 | - | - | - | - |
| Majority Baseline | 0.3333 | - | - | - | - |

### Additional Metrics

#### Accuracy
- **ResNet:** 0.6508 [0.6074, 0.6963]
- **LaBraM:** 0.6204 [0.5748, 0.6659]
- **CNN:** 0.6312 [0.5857, 0.6746]

#### Macro F1 Score
- **ResNet:** 0.6583 [0.6140, 0.7023]
- **LaBraM:** 0.6274 [0.5833, 0.6709]
- **CNN:** 0.6408 [0.5949, 0.6851]

### Per-Class Performance

#### ResNet (Best Overall)
- Class 0 (young): 64.9% recall
- Class 1 (middle): 61.5% recall
- Class 2 (old): 69.9% recall

#### LaBraM
- Class 0 (young): 80.5% recall (best for this class)
- Class 1 (middle): 50.0% recall
- Class 2 (old): 56.4% recall

#### CNN
- Class 0 (young): 58.4% recall
- Class 1 (middle): 72.4% recall (best for this class)
- Class 2 (old): 56.4% recall

---

## Statistical Significance

### Bootstrap Analysis
- **All models significantly outperform random baseline:**
  - P(delta > 0) = 1.0000 for all models and metrics
  - 95% CI for delta excludes 0 for all models
  - This indicates that in 100% of bootstrap samples, models performed better than random

### Permutation Test
- **All models:** p < 0.001 (permutation test with 1,000 replicates)
- This confirms that model predictions are significantly associated with true labels at the subject level

---

## Key Findings

1. **ResNet performs best** across all metrics:
   - Highest balanced accuracy (0.6545)
   - Most consistent performance across classes
   - Largest improvement over random baseline (Δ = 0.3712)

2. **All models significantly outperform baselines:**
   - All models show substantial improvement over random (Δ > 0.31)
   - All models outperform majority baseline
   - Statistical significance confirmed via bootstrap and permutation tests

3. **Class-specific strengths:**
   - LaBraM: Best at identifying young participants (Class 0: 80.5% recall)
   - CNN: Best at identifying middle-age participants (Class 1: 72.4% recall)
   - ResNet: Best overall balance and best at identifying older participants (Class 2: 69.9% recall)

4. **Uncertainty quantification:**
   - All 95% confidence intervals are reasonably tight
   - Bootstrap analysis confirms robust performance across resampling

---

## Conclusions

All three deep learning models (CNN, ResNet, LaBraM) demonstrate statistically significant performance in EEG-based age-group classification at the subject level. The ResNet-34 model achieves the best overall performance with a balanced accuracy of 65.5%, representing a 37% improvement over a stratified random baseline. The evaluation methodology ensures proper statistical inference by aggregating predictions at the participant level and using subject-level bootstrap for uncertainty estimation.

The results indicate that:
- Deep learning models can effectively learn age-related patterns from EEG data
- Subject-level evaluation is crucial for proper statistical assessment
- All models provide meaningful improvements over baseline methods
- Performance is consistent across different model architectures

---

## Technical Details

- **Bootstrap Replicates:** 2,000
- **Permutation Test Replicates:** 1,000
- **Random Seed:** 123
- **Evaluation Framework:** Subject-level aggregation with mean probability voting
- **Confidence Level:** 95%

---

*Report generated from evaluation results: `eval_subject_bootstrap/outputs/age_evaluation_4s_20260116_182211/`*
