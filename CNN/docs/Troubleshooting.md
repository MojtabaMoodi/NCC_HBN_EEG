# Troubleshooting

## Gender / class collapse (majority class only)

### What the results showed

- **Participant (majority vote):** 65.08% accuracy  
- **Participant (mean probability):** 65.08% accuracy  
- **Segment-level:** 64.75% accuracy  

The two participant-level methods gave **identical** accuracy.

### Why they are the same

Both methods assign **one prediction per participant** from that participant’s segments:

1. **Mean probability:** For each participant, average the model’s **probabilities** over all their segments, then take `argmax` → one class per participant.
2. **Majority vote:** For each participant, take the **predicted class** for each segment (`argmax` per segment), then take the **mode** → one class per participant.

If the model predicts the **same class for (almost) every segment** (e.g. always Female), then for every participant both methods yield the same prediction, hence the same accuracy.

### What “predicting female for everyone” means

From the confusion matrix (participant-level): if all participants are predicted as Female (300 correct, 161 wrong), the model has **collapsed to the majority class** and is not discriminating Male vs Female.

### Fixes

#### 1. Class weights (automatic by default)

The framework **computes class weights from the training data** (inverse frequency) for gender/age when you are not using oversampling.

- **Default:** For gender/age, if `balance_method` is not `"oversample"` and you don’t set `class_weight`, weights are computed from the train HDF5 and applied in the loss.
- **Stronger minority weighting:** Set `class_weight_power` in `TrainingConfig` (e.g. 1.5).
- **Manual override:** Set `class_weight=[1.0, 2.0]` (e.g. [Female, Male]) in config; adjust for your imbalance.

#### 2. Balanced sampling (CLI)

Use `--balance_method` so each **batch** is balanced:

- **Stratified** (`--balance_method stratified`): Each batch has (roughly) equal counts per class; participant round-robin within class. Class weights can still be used.
- **Oversample** (`--balance_method oversample`): Sampling with replacement (inverse class frequency); class weights are not set (balance via data).

```bash
python CNN/main.py --mode 4s --balance_method stratified --results_dir CNN_4s_stratified
python CNN/main.py --mode 4s --balance_method oversample --results_dir CNN_4s_oversample
```

When using stratified or oversample, the trainer does not abort on temporary model collapse; validation is also allowed to recover (warn and reset instead of failing the run).

#### 3. Label smoothing

Use a small label smoothing (e.g. 0.1) in `CrossEntropyLoss` so the model is less overconfident and less likely to collapse to one class.

#### 4. Learning rate and training

- Use a **lower learning rate** (e.g. 1e-4 or 5e-5) to avoid early collapse.
- Monitor **validation accuracy** and **per-class recall**; if the model never predicts the minority class, adjust class weight, sampling, or LR.

#### 5. Data and task

- Check **gender labels** and segment–participant mapping.
- Confirm that **EEG** carries usable signal for gender in your dataset.

### Summary

Majority vote and mean probability matched because the model predicted the same class for everyone (class collapse). The framework now computes **class weights** from train data by default for gender/age (when not using oversample). Using **`--balance_method stratified`** or **`--balance_method oversample`** (and optionally label smoothing) and retraining should help the model learn both classes.
