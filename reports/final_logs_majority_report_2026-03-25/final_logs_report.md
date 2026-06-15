# Final logs report
- **Generated**: 2026-03-25
- **Source**: `final_logs_majority`
- **Experiments**: total=29, success=29, failed=0
- **Coverage**: 4 `experiment_summary.json` → 10 rows; `reports/experiment_results.csv` → 13 rows; LaBraM `log.txt` 0, `output.log` 6 (total LaBraM rows: 6)
- **Target types**: `age`, `gender`
- **Model types**: `age_cnn`, `age_regression_cnn`, `age_resnet`, `age_resnet34`, `gender_cnn`, `gender_resnet`, `gender_resnet34`, `gender_resnet50`, `labram`
- **Task kinds**: `classification`, `regression`

## Leaderboards

### Gender (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_gender_2s` | `labram` | 0.9025 | — | — | — | — | — |
| `labram_gender_1s` | `labram` | 0.8919 | — | — | — | — | — |
| `gender_resnet18_2s` | `gender_resnet` | 0.8771 | 0.8791 | — | 0.8806 | 0.8726 | [link](final_logs_majority/RESNET/gender_resnet18_2s/reports/experiment_report.html) |
| `gender_resnet34_2s` | `gender_resnet34` | 0.8750 | 0.8774 | — | 0.8889 | 0.8769 | [link](final_logs_majority/RESNET/gender_resnet34_2s/reports/experiment_report.html) |
| `labram_gender_4s` | `labram` | 0.8729 | — | — | — | — | — |
| `gender_resnet50_2s` | `gender_resnet50` | 0.8708 | 0.8731 | — | 0.8806 | 0.8684 | [link](final_logs_majority/RESNET/gender_resnet50_2s/reports/experiment_report.html) |
| `gender_resnet50_1s` | `gender_resnet50` | 0.8686 | 0.8707 | — | 0.8694 | 0.8747 | [link](final_logs_majority/RESNET/gender_resnet50_1s/reports/experiment_report.html) |
| `gender_baseline_1s` | `gender_cnn` | 0.8644 | 0.8665 | 0.9410 | 0.8861 | 0.8684 | [link](final_logs_majority/CNN/cnn_eval_1s/reports/experiment_report.html) |
| `gender_resnet18_4s` | `gender_resnet` | 0.8623 | 0.8644 | — | 0.8667 | 0.8620 | [link](final_logs_majority/RESNET/gender_resnet18_4s/reports/experiment_report.html) |
| `gender_baseline_4s` | `gender_cnn` | 0.8602 | 0.8624 | 0.9419 | 0.8583 | 0.8577 | [link](final_logs_majority/CNN/cnn_eval_4s/reports/experiment_report.html) |
| `gender_resnet34_4s` | `gender_resnet34` | 0.8581 | 0.8610 | — | 0.8500 | 0.8556 | [link](final_logs_majority/RESNET/gender_resnet34_4s/reports/experiment_report.html) |
| `gender_resnet50_4s` | `gender_resnet50` | 0.8559 | 0.8587 | — | 0.8361 | 0.8493 | [link](final_logs_majority/RESNET/gender_resnet50_4s/reports/experiment_report.html) |
| `gender_resnet34_1s` | `gender_resnet34` | 0.8517 | 0.8545 | — | 0.8556 | 0.8556 | [link](final_logs_majority/RESNET/gender_resnet34_1s/reports/experiment_report.html) |
| `gender_baseline_2s` | `gender_cnn` | 0.8411 | 0.8445 | 0.9406 | 0.8583 | 0.8450 | [link](final_logs_majority/CNN/cnn_eval_2s/reports/experiment_report.html) |

#### Gender — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `gender_cnn` | 0.8644 (gender_baseline_1s) | 0.8411 (gender_baseline_2s) | 0.8602 (gender_baseline_4s) |
| `gender_resnet` | — | 0.8771 (gender_resnet18_2s) | 0.8623 (gender_resnet18_4s) |
| `gender_resnet34` | 0.8517 (gender_resnet34_1s) | 0.8750 (gender_resnet34_2s) | 0.8581 (gender_resnet34_4s) |
| `gender_resnet50` | 0.8686 (gender_resnet50_1s) | 0.8708 (gender_resnet50_2s) | 0.8559 (gender_resnet50_4s) |
| `labram` | 0.8919 (labram_gender_1s) | 0.9025 (labram_gender_2s) | 0.8729 (labram_gender_4s) |

### Age (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_age_1s` | `labram` | 0.6780 | — | — | — | — | — |
| `labram_age_2s` | `labram` | 0.6716 | — | — | — | — | — |
| `labram_age_4s` | `labram` | 0.6716 | — | — | — | — | — |
| `age_resnet18_2s` | `age_resnet` | 0.6038 | 0.5557 | — | 0.5889 | 0.6157 | [link](final_logs_majority/RESNET/age_resnet18_2s/reports/experiment_report.html) |
| `age_classification_4s` | `age_cnn` | 0.5636 | 0.4351 | 0.8631 | 0.5528 | 0.5626 | [link](final_logs_majority/CNN/cnn_eval_4s/reports/experiment_report.html) |
| `age_classification_1s` | `age_cnn` | 0.5572 | 0.4236 | 0.8436 | 0.5417 | 0.5520 | [link](final_logs_majority/CNN/cnn_eval_1s/reports/experiment_report.html) |
| `age_resnet18_4s` | `age_resnet` | 0.5530 | 0.4297 | — | 0.5250 | 0.5520 | [link](final_logs_majority/RESNET/age_resnet18_4s/reports/experiment_report.html) |
| `age_classification_2s` | `age_cnn` | 0.5508 | 0.4193 | 0.8547 | 0.5444 | 0.5520 | [link](final_logs_majority/CNN/cnn_eval_2s/reports/experiment_report.html) |
| `age_resnet34_1s` | `age_resnet34` | 0.5381 | 0.4099 | — | 0.5306 | 0.5414 | [link](final_logs_majority/RESNET/age_resnet34_1s/reports/experiment_report.html) |
| `age_resnet18_1s` | `age_resnet` | 0.5318 | 0.4041 | — | 0.5278 | 0.5308 | — |
| `age_resnet34_2s` | `age_resnet34` | 0.5318 | 0.4030 | — | 0.5278 | 0.5329 | [link](final_logs_majority/RESNET/age_resnet34_2s/reports/experiment_report.html) |
| `age_resnet34_4s` | `age_resnet34` | 0.5275 | 0.3983 | — | 0.5167 | 0.5287 | [link](final_logs_majority/RESNET/age_resnet34_4s/reports/experiment_report.html) |

#### Age — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `age_cnn` | 0.5572 (age_classification_1s) | 0.5508 (age_classification_2s) | 0.5636 (age_classification_4s) |
| `age_resnet` | 0.5318 (age_resnet18_1s) | 0.6038 (age_resnet18_2s) | 0.5530 (age_resnet18_4s) |
| `age_resnet34` | 0.5381 (age_resnet34_1s) | 0.5318 (age_resnet34_2s) | 0.5275 (age_resnet34_4s) |
| `labram` | 0.6780 (labram_age_1s) | 0.6716 (labram_age_2s) | 0.6716 (labram_age_4s) |

### Regression — all experiments, sorted by MAE (ascending)

| experiment | model | mae | rmse | r2 | active_mae | passive_mae | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `age_regression_4s` | `age_regression_cnn` | 1.6321 | 2.2256 | 0.6011 | 1.6224 | 1.6528 | [link](final_logs_majority/CNN/cnn_eval_4s/reports/experiment_report.html) |
| `age_regression_2s` | `age_regression_cnn` | 1.6906 | 2.3089 | 0.5707 | 1.6741 | 1.7106 | [link](final_logs_majority/CNN/cnn_eval_2s/reports/experiment_report.html) |
| `age_regression_1s` | `age_regression_cnn` | 1.7297 | 2.3200 | 0.5665 | 1.7072 | 1.7532 | [link](final_logs_majority/CNN/cnn_eval_1s/reports/experiment_report.html) |

## Full index

| experiment | target | kind | success | primary | active | passive | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `age_classification_1s` | `age` | `classification` | ✅ | 0.5572 | 0.5417 | 0.5520 | `final_logs_majority/CNN/cnn_eval_1s/experiment_summary.json` |
| `age_classification_2s` | `age` | `classification` | ✅ | 0.5508 | 0.5444 | 0.5520 | `final_logs_majority/CNN/cnn_eval_2s/experiment_summary.json` |
| `age_classification_4s` | `age` | `classification` | ✅ | 0.5636 | 0.5528 | 0.5626 | `final_logs_majority/CNN/cnn_eval_4s/experiment_summary.json` |
| `age_regression_1s` | `age` | `regression` | ✅ | 1.7297 | 1.7072 | 1.7532 | `final_logs_majority/CNN/cnn_eval_1s/experiment_summary.json` |
| `age_regression_2s` | `age` | `regression` | ✅ | 1.6906 | 1.6741 | 1.7106 | `final_logs_majority/CNN/cnn_eval_2s/experiment_summary.json` |
| `age_regression_4s` | `age` | `regression` | ✅ | 1.6321 | 1.6224 | 1.6528 | `final_logs_majority/CNN/cnn_eval_4s/experiment_summary.json` |
| `age_resnet18_1s` | `age` | `classification` | ✅ | 0.5318 | 0.5278 | 0.5308 | `final_logs_majority/experiment_summary.json` |
| `age_resnet18_2s` | `age` | `classification` | ✅ | 0.6038 | 0.5889 | 0.6157 | `final_logs_majority/RESNET/age_resnet18_2s/reports/experiment_results.csv` |
| `age_resnet18_4s` | `age` | `classification` | ✅ | 0.5530 | 0.5250 | 0.5520 | `final_logs_majority/RESNET/age_resnet18_4s/reports/experiment_results.csv` |
| `age_resnet34_1s` | `age` | `classification` | ✅ | 0.5381 | 0.5306 | 0.5414 | `final_logs_majority/RESNET/age_resnet34_1s/reports/experiment_results.csv` |
| `age_resnet34_2s` | `age` | `classification` | ✅ | 0.5318 | 0.5278 | 0.5329 | `final_logs_majority/RESNET/age_resnet34_2s/reports/experiment_results.csv` |
| `age_resnet34_4s` | `age` | `classification` | ✅ | 0.5275 | 0.5167 | 0.5287 | `final_logs_majority/RESNET/age_resnet34_4s/reports/experiment_results.csv` |
| `labram_age_1s` | `age` | `classification` | ✅ | 0.6780 | — | — | `final_logs_majority/LaBraM/labram_age_1s/output.log` |
| `labram_age_2s` | `age` | `classification` | ✅ | 0.6716 | — | — | `final_logs_majority/LaBraM/labram_age_2s/output.log` |
| `labram_age_4s` | `age` | `classification` | ✅ | 0.6716 | — | — | `final_logs_majority/LaBraM/labram_age_4s/output.log` |
| `gender_baseline_1s` | `gender` | `classification` | ✅ | 0.8644 | 0.8861 | 0.8684 | `final_logs_majority/CNN/cnn_eval_1s/experiment_summary.json` |
| `gender_baseline_2s` | `gender` | `classification` | ✅ | 0.8411 | 0.8583 | 0.8450 | `final_logs_majority/CNN/cnn_eval_2s/experiment_summary.json` |
| `gender_baseline_4s` | `gender` | `classification` | ✅ | 0.8602 | 0.8583 | 0.8577 | `final_logs_majority/CNN/cnn_eval_4s/experiment_summary.json` |
| `gender_resnet18_2s` | `gender` | `classification` | ✅ | 0.8771 | 0.8806 | 0.8726 | `final_logs_majority/RESNET/gender_resnet18_2s/reports/experiment_results.csv` |
| `gender_resnet18_4s` | `gender` | `classification` | ✅ | 0.8623 | 0.8667 | 0.8620 | `final_logs_majority/RESNET/gender_resnet18_4s/reports/experiment_results.csv` |
| `gender_resnet34_1s` | `gender` | `classification` | ✅ | 0.8517 | 0.8556 | 0.8556 | `final_logs_majority/RESNET/gender_resnet34_1s/reports/experiment_results.csv` |
| `gender_resnet34_2s` | `gender` | `classification` | ✅ | 0.8750 | 0.8889 | 0.8769 | `final_logs_majority/RESNET/gender_resnet34_2s/reports/experiment_results.csv` |
| `gender_resnet34_4s` | `gender` | `classification` | ✅ | 0.8581 | 0.8500 | 0.8556 | `final_logs_majority/RESNET/gender_resnet34_4s/reports/experiment_results.csv` |
| `gender_resnet50_1s` | `gender` | `classification` | ✅ | 0.8686 | 0.8694 | 0.8747 | `final_logs_majority/RESNET/gender_resnet50_1s/reports/experiment_results.csv` |
| `gender_resnet50_2s` | `gender` | `classification` | ✅ | 0.8708 | 0.8806 | 0.8684 | `final_logs_majority/RESNET/gender_resnet50_2s/reports/experiment_results.csv` |
| `gender_resnet50_4s` | `gender` | `classification` | ✅ | 0.8559 | 0.8361 | 0.8493 | `final_logs_majority/RESNET/gender_resnet50_4s/reports/experiment_results.csv` |
| `labram_gender_1s` | `gender` | `classification` | ✅ | 0.8919 | — | — | `final_logs_majority/LaBraM/labram_gender_1s/output.log` |
| `labram_gender_2s` | `gender` | `classification` | ✅ | 0.9025 | — | — | `final_logs_majority/LaBraM/labram_gender_2s/output.log` |
| `labram_gender_4s` | `gender` | `classification` | ✅ | 0.8729 | — | — | `final_logs_majority/LaBraM/labram_gender_4s/output.log` |

## Missing experiments (expected full grid)

This tree is expected to contain **33** distinct experiment names: **9** CNN runs (`gender_baseline_*`, `age_classification_*`, `age_regression_*` at 1s/2s/4s), **18** ResNet runs (three backbones × three window lengths × age and gender), **6** LaBraM runs (`labram_age_*`, `labram_gender_*`). The count is **33** (not 34) because `age_resnet18_1s` is listed once (root `experiment_summary.json` and `RESNET/age_resnet18_1s/` are the same experiment).

- **Present in this report**: 29 rows  
- **Expected names**: 33  

The following expected names have **no** ingested row (no `experiment_summary.json` entry, no row in `reports/experiment_results.csv`, and no LaBraM `log.txt` / `output.log`):

- `age_resnet50_1s`
- `age_resnet50_2s`
- `age_resnet50_4s`
- `gender_resnet18_1s`
