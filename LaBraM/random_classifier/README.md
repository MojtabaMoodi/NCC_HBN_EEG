# Random classifier baseline (age / gender)

Segment-level random predictions on the **test** split, with the same metrics and
participant aggregation as LaBraM eval (`--aggregate_by_participant majority_vote`).

Implementation:

| File | Role |
|------|------|
| `../eval_random_classifier.py` | CLI |
| `../random_classifier.py` | Label sampling strategies |
| `../classification_eval_utils.py` | Shared metrics + participant aggregation |

## Results directory

JSON outputs are written under **`LaBraM/random_classifier/results/`**:

```
LaBraM/random_classifier/
├── README.md
├── run_train_segment_frequency_eval.sh
└── results/
    ├── age_1s_train_segment_frequency.json
    ├── age_2s_train_segment_frequency.json
    ├── age_4s_train_segment_frequency.json
    ├── gender_1s_train_segment_frequency.json
    ├── gender_2s_train_segment_frequency.json
    └── gender_4s_train_segment_frequency.json
```

Compare **`test.participant_metrics`** in each JSON to LaBraM
`participant (majority vote)` in `final_logs_majority/LaBraM/labram_{age,gender}_*/output.log`.

## Random strategy: `train_segment_frequency`

Labels are sampled **independently per test segment** using class proportions from
**training-split segment counts** (same HDF5 metadata scan as
`compute_class_weights_from_train_hdf5`). Respects `--task_type` (`both`, `active`, or
`passive`).

Alternative: `--random_strategy uniform` (equal class probability; LaBraM `1/nb_classes`).

## Run all age + gender baselines (recommended)

From the **EEG project root**, with `eeg_env` active:

```bash
bash LaBraM/random_classifier/run_train_segment_frequency_eval.sh
```

Optional environment overrides (same semantics as LaBraM eval):

| Variable | Example |
|----------|---------|
| `DATA_PATH` | `/home/mojtabam/scratch/processed_eeg_data_hdf5` |
| `RANDOM_SEED` | `42` |

## Individual commands

Replace `{age\|gender}` and `{1s\|2s\|4s}` as needed. Use `--num_workers 0` for **1s**,
`4` for **2s** and **4s** (matches LaBraM eval).

```bash
python LaBraM/eval_random_classifier.py \
  --dataset {age_baseline|gender_baseline} \
  --segment_length {1s|2s|4s} \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --random_seed 42 \
  --random_strategy train_segment_frequency \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers {0|4} \
  --output_json LaBraM/random_classifier/results/{age|gender}_{1s|2s|4s}_train_segment_frequency.json
```

Examples:

Age, 1s (`--num_workers 0`, same as LaBraM eval for 1s HDF5):

```bash
python LaBraM/eval_random_classifier.py \
  --dataset age_baseline \
  --segment_length 1s \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --random_seed 42 \
  --random_strategy train_segment_frequency \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers 0 \
  --output_json LaBraM/random_classifier/results/age_1s_train_segment_frequency.json
```

Age, 4s:

```bash
python LaBraM/eval_random_classifier.py \
  --dataset age_baseline \
  --segment_length 4s \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --random_seed 42 \
  --random_strategy train_segment_frequency \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers 4 \
  --output_json LaBraM/random_classifier/results/age_4s_train_segment_frequency.json
```

Gender, 2s:

```bash
python LaBraM/eval_random_classifier.py \
  --dataset gender_baseline \
  --segment_length 2s \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --random_seed 42 \
  --random_strategy train_segment_frequency \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers 4 \
  --output_json LaBraM/random_classifier/results/gender_2s_train_segment_frequency.json
```

Optional: add `--include_task_type_breakdown` for separate active/passive metrics
(not shown in the main LaBraM final-logs table).

## Metrics in each JSON

- **Age:** `accuracy`, `balanced_accuracy`, `f1_weighted` (segment, participant mean-prob, participant majority vote)
- **Gender:** `accuracy`, `balanced_accuracy`, `pr_auc`, `roc_auc`, `f1_weighted`
- **`class_proportions`:** train segment frequencies used for sampling
