# Final logs report
- **Generated**: 2026-05-27
- **Source**: `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen`
- **Experiments**: total=9, success=9, failed=0
- **Coverage**: 0 `experiment_summary.json` → 0 rows; `reports/experiment_results.csv` → 0 rows; LaBraM `log.txt` 9, `output.log` 0 (total LaBraM rows: 9)
- **Target types**: `age`, `gender`
- **Model types**: `labram`
- **Task kinds**: `classification`, `regression`

## Leaderboards

### Gender (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_gender_frozen_4s` | `labram` | 0.6149 | 0.5716 | 0.6823 | 0.6209 | 0.5879 | — |
| `labram_gender_frozen_2s` | `labram` | 0.6133 | 0.5358 | 0.6600 | 0.6521 | 0.5920 | — |
| `labram_gender_frozen_1s` | `labram` | 0.5989 | 0.5333 | 0.6388 | 0.6268 | 0.5729 | — |

#### Gender — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `labram` | 0.5989 (labram_gender_frozen_1s) | 0.6133 (labram_gender_frozen_2s) | 0.6149 (labram_gender_frozen_4s) |

### Age (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_age_frozen_4s` | `labram` | 0.4433 | 0.4212 | — | 0.4620 | 0.4489 | — |
| `labram_age_frozen_2s` | `labram` | 0.4305 | 0.3665 | — | 0.4534 | 0.4368 | — |
| `labram_age_frozen_1s` | `labram` | 0.4266 | 0.3969 | — | 0.4450 | 0.4253 | — |

#### Age — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `labram` | 0.4266 (labram_age_frozen_1s) | 0.4305 (labram_age_frozen_2s) | 0.4433 (labram_age_frozen_4s) |

### Regression — all experiments, sorted by MAE (ascending)

| experiment | model | mae | rmse | r2 | active_mae | passive_mae | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_age_regression_frozen_4s` | `labram` | 2.6038 | 3.2256 | 0.0933 | 2.6811 | 2.6793 | — |
| `labram_age_regression_frozen_2s` | `labram` | 2.6135 | 3.2590 | 0.0761 | 2.6906 | 2.6886 | — |
| `labram_age_regression_frozen_1s` | `labram` | 2.6513 | 3.2812 | 0.0633 | 2.6954 | 2.6954 | — |

## Full index

| experiment | target | kind | success | primary | active | passive | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_age_frozen_1s` | `age` | `classification` | ✅ | 0.4266 | 0.4450 | 0.4253 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_1s/output/log.txt` |
| `labram_age_frozen_2s` | `age` | `classification` | ✅ | 0.4305 | 0.4534 | 0.4368 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_2s/output/log.txt` |
| `labram_age_frozen_4s` | `age` | `classification` | ✅ | 0.4433 | 0.4620 | 0.4489 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_4s/output/log.txt` |
| `labram_age_regression_frozen_1s` | `age` | `regression` | ✅ | 2.6513 | 2.6954 | 2.6954 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_regression_frozen_1s/output/log.txt` |
| `labram_age_regression_frozen_2s` | `age` | `regression` | ✅ | 2.6135 | 2.6906 | 2.6886 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_regression_frozen_2s/output/log.txt` |
| `labram_age_regression_frozen_4s` | `age` | `regression` | ✅ | 2.6038 | 2.6811 | 2.6793 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_regression_frozen_4s/output/log.txt` |
| `labram_gender_frozen_1s` | `gender` | `classification` | ✅ | 0.5989 | 0.6268 | 0.5729 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_1s/output/log.txt` |
| `labram_gender_frozen_2s` | `gender` | `classification` | ✅ | 0.6133 | 0.6521 | 0.5920 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_2s/output/log.txt` |
| `labram_gender_frozen_4s` | `gender` | `classification` | ✅ | 0.6149 | 0.6209 | 0.5879 | `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_4s/output/log.txt` |

## Parsing warnings

- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_1s/train_resume_slurm-3728826.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_2s/train.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_age_frozen_4s/train.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_1s/train_resume_slurm-3728827.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_2s/train.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
- /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/LaBraM/labram_gender_frozen_4s/train.log: ACTIVE/PASSIVE lines have n=0 sample counts; using JSONL test_accuracy for overall and parsed active/passive accuracies
