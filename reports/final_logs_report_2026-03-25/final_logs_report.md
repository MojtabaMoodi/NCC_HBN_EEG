# Final logs report
- **Generated**: 2026-03-25
- **Source**: `final_logs`
- **Experiments**: total=33, success=33, failed=0
- **Coverage**: 21 `experiment_summary.json` → 27 rows; `reports/experiment_results.csv` → 0 rows; LaBraM `log.txt` 6, `output.log` 0 (total LaBraM rows: 6)
- **Target types**: `age`, `gender`
- **Model types**: `age_cnn`, `age_regression_cnn`, `age_resnet`, `age_resnet34`, `age_resnet50`, `gender_cnn`, `gender_resnet`, `gender_resnet34`, `gender_resnet50`, `labram`
- **Task kinds**: `classification`, `regression`

## Leaderboards

### Gender (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_gender_4s` | `labram` | 0.8753 | — | — | 0.8951 | 0.8716 | — |
| `labram_gender_2s` | `labram` | 0.8728 | — | — | 0.8981 | 0.8685 | — |
| `labram_gender_1s` | `labram` | 0.8694 | — | — | 0.8931 | 0.8654 | — |
| `gender_baseline_1s` | `gender_cnn` | 0.8151 | 0.8183 | 0.9025 | 0.8497 | 0.8097 | [link](final_logs/CNN/CNN_1s/experiment_report.html) |
| `gender_resnet18_2s` | `gender_resnet` | 0.8113 | 0.8148 | 0.9016 | 0.8384 | 0.8066 | [link](final_logs/RESNET/RESNET18_gender_2s/experiment_report.html) |
| `gender_resnet18_4s` | `gender_resnet` | 0.8082 | 0.8112 | 0.8930 | 0.8317 | 0.8038 | [link](final_logs/RESNET/RESNET18_gender_4s/experiment_report.html) |
| `gender_baseline_4s` | `gender_cnn` | 0.8080 | 0.8111 | 0.8958 | 0.8346 | 0.8033 | [link](final_logs/CNN/CNN_4s/experiment_report.html) |
| `gender_resnet50_1s` | `gender_resnet50` | 0.8039 | 0.8074 | 0.8935 | 0.8338 | 0.7988 | [link](final_logs/RESNET/RESNET50_gender_1s/experiment_report.html) |
| `gender_resnet50_2s` | `gender_resnet50` | 0.8037 | 0.8076 | 0.9024 | 0.8304 | 0.7991 | [link](final_logs/RESNET/RESNET50_gender_2s/experiment_report.html) |
| `gender_resnet34_2s` | `gender_resnet34` | 0.8026 | 0.8066 | 0.9039 | 0.8273 | 0.7983 | [link](final_logs/RESNET/RESNET34_gender_2s/experiment_report.html) |
| `gender_baseline_2s` | `gender_cnn` | 0.8021 | 0.8062 | 0.9079 | 0.8390 | 0.7961 | [link](final_logs/CNN/CNN_2s/experiment_report.html) |
| `gender_resnet34_1s` | `gender_resnet34` | 0.7920 | 0.7961 | 0.8848 | 0.8195 | 0.7874 | [link](final_logs/RESNET/RESNET34_gender_1s/experiment_report.html) |
| `gender_resnet34_4s` | `gender_resnet34` | 0.7884 | 0.7927 | 0.8916 | 0.8094 | 0.7845 | [link](final_logs/RESNET/RESNET34_gender_4s/experiment_report.html) |
| `gender_resnet18_1s` | `gender_resnet` | 0.7847 | 0.7892 | 0.8882 | 0.8124 | 0.7800 | [link](final_logs/RESNET/RESNET18_gender_1s/experiment_report.html) |
| `gender_resnet50_4s` | `gender_resnet50` | 0.7820 | 0.7865 | 0.8890 | 0.8049 | 0.7778 | [link](final_logs/RESNET/RESNET50_gender_4s/experiment_report.html) |

#### Gender — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `gender_cnn` | 0.8151 (gender_baseline_1s) | 0.8021 (gender_baseline_2s) | 0.8080 (gender_baseline_4s) |
| `gender_resnet` | 0.7847 (gender_resnet18_1s) | 0.8113 (gender_resnet18_2s) | 0.8082 (gender_resnet18_4s) |
| `gender_resnet34` | 0.7920 (gender_resnet34_1s) | 0.8026 (gender_resnet34_2s) | 0.7884 (gender_resnet34_4s) |
| `gender_resnet50` | 0.8039 (gender_resnet50_1s) | 0.8037 (gender_resnet50_2s) | 0.7820 (gender_resnet50_4s) |
| `labram` | 0.8694 (labram_gender_1s) | 0.8728 (labram_gender_2s) | 0.8753 (labram_gender_4s) |

### Age (classification) — all experiments, sorted by accuracy (descending)

| experiment | model | acc | f1_w | auc | active_acc | passive_acc | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `labram_age_4s` | `labram` | 0.5927 | — | — | 0.5967 | 0.5920 | — |
| `labram_age_2s` | `labram` | 0.5730 | — | — | 0.5843 | 0.5711 | — |
| `age_classification_4s` | `age_cnn` | 0.5452 | 0.4934 | 0.7532 | 0.5354 | 0.5459 | [link](final_logs/CNN/CNN_4s/experiment_report.html) |
| `age_classification_2s` | `age_cnn` | 0.5348 | 0.4810 | 0.7420 | 0.5334 | 0.5351 | [link](final_logs/CNN/CNN_2s/experiment_report.html) |
| `age_resnet50_4s` | `age_resnet50` | 0.5307 | 0.4844 | 0.7432 | 0.5350 | 0.5299 | [link](final_logs/RESNET/RESNET50_age_4s/experiment_report.html) |
| `age_resnet18_2s` | `age_resnet` | 0.5274 | 0.5096 | 0.7200 | 0.5311 | 0.5268 | [link](final_logs/RESNET/RESNET18_age_2s/experiment_report.html) |
| `age_classification_1s` | `age_cnn` | 0.5209 | 0.4738 | 0.7231 | 0.5178 | 0.5202 | [link](final_logs/CNN/CNN_1s/experiment_report.html) |
| `age_resnet50_2s` | `age_resnet50` | 0.5196 | 0.4775 | 0.7255 | 0.5243 | 0.5188 | [link](final_logs/RESNET/RESNET50_age_2s/experiment_report.html) |
| `labram_age_1s` | `labram` | 0.5196 | — | — | 0.5352 | 0.5169 | — |
| `age_resnet18_4s` | `age_resnet` | 0.5113 | 0.4466 | 0.7285 | 0.4990 | 0.5136 | [link](final_logs/RESNET/RESNET18_age_42/experiment_report.html) |
| `age_resnet34_2s` | `age_resnet34` | 0.5062 | 0.4556 | 0.7187 | 0.5027 | 0.5068 | [link](final_logs/RESNET/RESNET34_age_2s/experiment_report.html) |
| `age_resnet34_4s` | `age_resnet34` | 0.4977 | 0.4442 | 0.7183 | 0.4954 | 0.4981 | [link](final_logs/RESNET/RESNET34_age_4s/experiment_report.html) |
| `age_resnet50_1s` | `age_resnet50` | 0.4829 | 0.4094 | 0.6933 | 0.4774 | 0.4838 | [link](final_logs/RESNET/RESNET50_age_1s/experiment_report.html) |
| `age_resnet34_1s` | `age_resnet34` | 0.4822 | 0.4160 | 0.6943 | 0.4787 | 0.4828 | [link](final_logs/RESNET/RESNET34_age_1s/experiment_report.html) |
| `age_resnet18_1s` | `age_resnet` | 0.4772 | 0.4168 | 0.6913 | 0.4773 | 0.4771 | [link](final_logs/RESNET/RESNET18_age_1s/experiment_report.html) |

#### Age — accuracy by window (1s / 2s / 4s)

| model | 1s | 2s | 4s |
| --- | --- | --- | --- |
| `age_cnn` | 0.5209 (age_classification_1s) | 0.5348 (age_classification_2s) | 0.5452 (age_classification_4s) |
| `age_resnet` | 0.4772 (age_resnet18_1s) | 0.5274 (age_resnet18_2s) | 0.5113 (age_resnet18_4s) |
| `age_resnet34` | 0.4822 (age_resnet34_1s) | 0.5062 (age_resnet34_2s) | 0.4977 (age_resnet34_4s) |
| `age_resnet50` | 0.4829 (age_resnet50_1s) | 0.5196 (age_resnet50_2s) | 0.5307 (age_resnet50_4s) |
| `labram` | 0.5196 (labram_age_1s) | 0.5730 (labram_age_2s) | 0.5927 (labram_age_4s) |

### Regression — all experiments, sorted by MAE (ascending)

| experiment | model | mae | rmse | r2 | active_mae | passive_mae | report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `age_regression_4s` | `age_regression_cnn` | 1.9975 | 2.6699 | 0.4168 | 1.9635 | 2.0039 | [link](final_logs/CNN/CNN_4s/experiment_report.html) |
| `age_regression_2s` | `age_regression_cnn` | 2.0010 | 2.6866 | 0.4065 | 1.9410 | 2.0144 | [link](final_logs/CNN/CNN_2s/experiment_report.html) |
| `age_regression_1s` | `age_regression_cnn` | 2.0829 | 2.7727 | 0.3678 | 2.0145 | 2.0957 | [link](final_logs/CNN/CNN_1s/experiment_report.html) |

## Full index

| experiment | target | kind | success | primary | active | passive | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `age_classification_1s` | `age` | `classification` | ✅ | 0.5209 | 0.5178 | 0.5202 | `final_logs/CNN/CNN_1s/experiment_summary.json` |
| `age_classification_2s` | `age` | `classification` | ✅ | 0.5348 | 0.5334 | 0.5351 | `final_logs/CNN/CNN_2s/experiment_summary.json` |
| `age_classification_4s` | `age` | `classification` | ✅ | 0.5452 | 0.5354 | 0.5459 | `final_logs/CNN/CNN_4s/experiment_summary.json` |
| `age_regression_1s` | `age` | `regression` | ✅ | 2.0829 | 2.0145 | 2.0957 | `final_logs/CNN/CNN_1s/experiment_summary.json` |
| `age_regression_2s` | `age` | `regression` | ✅ | 2.0010 | 1.9410 | 2.0144 | `final_logs/CNN/CNN_2s/experiment_summary.json` |
| `age_regression_4s` | `age` | `regression` | ✅ | 1.9975 | 1.9635 | 2.0039 | `final_logs/CNN/CNN_4s/experiment_summary.json` |
| `age_resnet18_1s` | `age` | `classification` | ✅ | 0.4772 | 0.4773 | 0.4771 | `final_logs/RESNET/RESNET18_age_1s/experiment_summary.json` |
| `age_resnet18_2s` | `age` | `classification` | ✅ | 0.5274 | 0.5311 | 0.5268 | `final_logs/RESNET/RESNET18_age_2s/experiment_summary.json` |
| `age_resnet18_4s` | `age` | `classification` | ✅ | 0.5113 | 0.4990 | 0.5136 | `final_logs/RESNET/RESNET18_age_42/experiment_summary.json` |
| `age_resnet34_1s` | `age` | `classification` | ✅ | 0.4822 | 0.4787 | 0.4828 | `final_logs/RESNET/RESNET34_age_1s/experiment_summary.json` |
| `age_resnet34_2s` | `age` | `classification` | ✅ | 0.5062 | 0.5027 | 0.5068 | `final_logs/RESNET/RESNET34_age_2s/experiment_summary.json` |
| `age_resnet34_4s` | `age` | `classification` | ✅ | 0.4977 | 0.4954 | 0.4981 | `final_logs/RESNET/RESNET34_age_4s/experiment_summary.json` |
| `age_resnet50_1s` | `age` | `classification` | ✅ | 0.4829 | 0.4774 | 0.4838 | `final_logs/RESNET/RESNET50_age_1s/experiment_summary.json` |
| `age_resnet50_2s` | `age` | `classification` | ✅ | 0.5196 | 0.5243 | 0.5188 | `final_logs/RESNET/RESNET50_age_2s/experiment_summary.json` |
| `age_resnet50_4s` | `age` | `classification` | ✅ | 0.5307 | 0.5350 | 0.5299 | `final_logs/RESNET/RESNET50_age_4s/experiment_summary.json` |
| `labram_age_1s` | `age` | `classification` | ✅ | 0.5196 | 0.5352 | 0.5169 | `final_logs/LaBraM/labram_age_1s/log.txt` |
| `labram_age_2s` | `age` | `classification` | ✅ | 0.5730 | 0.5843 | 0.5711 | `final_logs/LaBraM/labram_age_2s/log.txt` |
| `labram_age_4s` | `age` | `classification` | ✅ | 0.5927 | 0.5967 | 0.5920 | `final_logs/LaBraM/labram_age_4s/log.txt` |
| `gender_baseline_1s` | `gender` | `classification` | ✅ | 0.8151 | 0.8497 | 0.8097 | `final_logs/CNN/CNN_1s/experiment_summary.json` |
| `gender_baseline_2s` | `gender` | `classification` | ✅ | 0.8021 | 0.8390 | 0.7961 | `final_logs/CNN/CNN_2s/experiment_summary.json` |
| `gender_baseline_4s` | `gender` | `classification` | ✅ | 0.8080 | 0.8346 | 0.8033 | `final_logs/CNN/CNN_4s/experiment_summary.json` |
| `gender_resnet18_1s` | `gender` | `classification` | ✅ | 0.7847 | 0.8124 | 0.7800 | `final_logs/RESNET/RESNET18_gender_1s/experiment_summary.json` |
| `gender_resnet18_2s` | `gender` | `classification` | ✅ | 0.8113 | 0.8384 | 0.8066 | `final_logs/RESNET/RESNET18_gender_2s/experiment_summary.json` |
| `gender_resnet18_4s` | `gender` | `classification` | ✅ | 0.8082 | 0.8317 | 0.8038 | `final_logs/RESNET/RESNET18_gender_4s/experiment_summary.json` |
| `gender_resnet34_1s` | `gender` | `classification` | ✅ | 0.7920 | 0.8195 | 0.7874 | `final_logs/RESNET/RESNET34_gender_1s/experiment_summary.json` |
| `gender_resnet34_2s` | `gender` | `classification` | ✅ | 0.8026 | 0.8273 | 0.7983 | `final_logs/RESNET/RESNET34_gender_2s/experiment_summary.json` |
| `gender_resnet34_4s` | `gender` | `classification` | ✅ | 0.7884 | 0.8094 | 0.7845 | `final_logs/RESNET/RESNET34_gender_4s/experiment_summary.json` |
| `gender_resnet50_1s` | `gender` | `classification` | ✅ | 0.8039 | 0.8338 | 0.7988 | `final_logs/RESNET/RESNET50_gender_1s/experiment_summary.json` |
| `gender_resnet50_2s` | `gender` | `classification` | ✅ | 0.8037 | 0.8304 | 0.7991 | `final_logs/RESNET/RESNET50_gender_2s/experiment_summary.json` |
| `gender_resnet50_4s` | `gender` | `classification` | ✅ | 0.7820 | 0.8049 | 0.7778 | `final_logs/RESNET/RESNET50_gender_4s/experiment_summary.json` |
| `labram_gender_1s` | `gender` | `classification` | ✅ | 0.8694 | 0.8931 | 0.8654 | `final_logs/LaBraM/labram_gender_1s/log.txt` |
| `labram_gender_2s` | `gender` | `classification` | ✅ | 0.8728 | 0.8981 | 0.8685 | `final_logs/LaBraM/labram_gender_2s/log.txt` |
| `labram_gender_4s` | `gender` | `classification` | ✅ | 0.8753 | 0.8951 | 0.8716 | `final_logs/LaBraM/labram_gender_4s/log.txt` |
