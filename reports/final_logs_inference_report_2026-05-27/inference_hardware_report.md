# Inference time, parameters, and hardware — `final_logs`

Generated from `final_logs` on 2026-05-27.

## Notes

- **LaBraM** logs report per-batch inference as `s/it` from final `Test` / `Active Test` / `Passive Test` blocks.
- **CNN / ResNet** logs report pooled-test `Evaluation time` and total eval time (including active/passive splits). Per-sample ms uses pooled time for combined test; active/passive ms uses the remainder split by sample counts.
- **Parameter counts**: LaBraM from training logs; CNN from model architecture; ResNet from saved checkpoints.
- All runs in this report used **NVIDIA L40S** GPUs (count varies by experiment).

## Age Classification

### LaBraM

| Segment | Params | GPU | #GPUs | Test s/batch | Active s/batch | Passive s/batch | Source |
|---------|--------|-----|-------|--------------|----------------|-----------------|--------|
| 1s | 5,825,339 | NVIDIA L40S | 2 | 1.3121 | 0.0732 | 0.0662 | `final_logs/LaBraM/labram_age_1s/train3.log` |
| 2s | 5,825,339 | NVIDIA L40S | 4 | 0.2734 | 0.2092 | 0.1936 | `final_logs/LaBraM/labram_age_2s/train.log` |
| 4s | 5,825,339 | NVIDIA L40S | 4 | 0.2670 | 0.2229 | 0.1994 | `final_logs/LaBraM/labram_age_4s/train.log` |

### CNN

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| eeg_cnn | 1s | 593,522 | NVIDIA L40S | 4 | 1199.5 | 0.641 | 1.875 | 1.875 | `final_logs/CNN/CNN_1s/cnn_1s.log` |
| eeg_cnn | 2s | 593,522 | NVIDIA L40S | 4 | 560.1 | 0.715 | 1.624 | 1.624 | `final_logs/CNN/CNN_2s/cnn_2s.log` |
| eeg_cnn | 4s | 593,522 | NVIDIA L40S | 4 | 351.7 | 0.876 | 1.745 | 1.745 | `final_logs/CNN/CNN_4s/cnn_4s.log` |

### ResNet

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| resnet18 | 1s | 3,946,007 | NVIDIA L40S | 4 | 955.7 | 0.559 | 1.445 | 1.445 | `final_logs/RESNET/RESNET18_age_1s/resnet18_age_1s.log` |
| resnet18 | 2s | 3,946,007 | NVIDIA L40S | 4 | 544.5 | 0.694 | 1.580 | 1.580 | `final_logs/RESNET/RESNET18_age_2s/resnet18_age_2s.log` |
| resnet18 | 4s | 3,946,007 | NVIDIA L40S | 4 | 358.0 | 0.831 | 1.838 | 1.838 | `final_logs/RESNET/RESNET18_age_4s/resnet18_age_4s.log` |
| resnet34 | 1s | 7,327,783 | NVIDIA L40S | 4 | 916.6 | 0.545 | 1.378 | 1.378 | `final_logs/RESNET/RESNET34_age_1s/resnet34_age_1s.log` |
| resnet34 | 2s | 7,327,783 | NVIDIA L40S | 4 | 539.5 | 0.693 | 1.560 | 1.560 | `final_logs/RESNET/RESNET34_age_2s/resnet34_age_2s.log` |
| resnet34 | 4s | 7,327,783 | NVIDIA L40S | 4 | 275.4 | 0.647 | 1.405 | 1.405 | `final_logs/RESNET/RESNET34_age_4s/resnet34_age_4s.log` |
| resnet50 | 1s | 16,296,504 | NVIDIA L40S | 4 | 930.3 | 0.565 | 1.386 | 1.386 | `final_logs/RESNET/RESNET50_age_1s/resnet50_age_1s.log` |
| resnet50 | 2s | 16,296,504 | NVIDIA L40S | 4 | 417.7 | 0.506 | 1.238 | 1.238 | `final_logs/RESNET/RESNET50_age_2s/resnet50_age_2s.log` |
| resnet50 | 4s | 16,296,504 | NVIDIA L40S | 4 | 281.3 | 0.665 | 1.431 | 1.431 | `final_logs/RESNET/RESNET50_age_4s/resnet50_age_4s.log` |

## Gender Classification

### LaBraM

| Segment | Params | GPU | #GPUs | Test s/batch | Active s/batch | Passive s/batch | Source |
|---------|--------|-----|-------|--------------|----------------|-----------------|--------|
| 1s | 5,824,937 | NVIDIA L40S | 1 | 1.5129 | 0.0222 | 0.0184 | `final_logs/LaBraM/labram_gender_1s/train2.log` |
| 2s | 5,824,937 | NVIDIA L40S | 2 | 0.1069 | 0.0683 | 0.0658 | `final_logs/LaBraM/labram_gender_2s/eval_f1.log` |
| 4s | 5,824,937 | NVIDIA L40S | 2 | 0.1238 | 0.0943 | 0.0795 | `final_logs/LaBraM/labram_gender_4s/eval_f1.log` |

### CNN

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| eeg_cnn | 1s | 593,522 | NVIDIA L40S | 4 | 991.3 | 0.594 | 1.485 | 1.485 | `final_logs/CNN/CNN_1s/cnn_1s.log` |
| eeg_cnn | 2s | 593,522 | NVIDIA L40S | 4 | 557.1 | 0.741 | 1.585 | 1.585 | `final_logs/CNN/CNN_2s/cnn_2s.log` |
| eeg_cnn | 4s | 593,522 | NVIDIA L40S | 4 | 341.9 | 0.820 | 1.729 | 1.729 | `final_logs/CNN/CNN_4s/cnn_4s.log` |

### ResNet

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| resnet18 | 1s | 3,946,007 | NVIDIA L40S | 4 | 744.2 | 0.411 | 1.150 | 1.150 | `final_logs/RESNET/RESNET18_gender_1s/resnet18_gender_1s.log` |
| resnet18 | 2s | 3,946,007 | NVIDIA L40S | 4 | 457.7 | 0.549 | 1.362 | 1.362 | `final_logs/RESNET/RESNET18_gender_2s/resnet18_gender_2s.log` |
| resnet18 | 4s | 3,946,007 | NVIDIA L40S | 4 | 287.0 | 0.683 | 1.456 | 1.456 | `final_logs/RESNET/RESNET18_gender_4s/resnet18_gender_4s.log` |
| resnet34 | 1s | 7,327,783 | NVIDIA L40S | 4 | 782.0 | 0.443 | 1.197 | 1.197 | `final_logs/RESNET/RESNET34_gender_1s/resnet34_gender_1s.log` |
| resnet34 | 2s | 7,327,783 | NVIDIA L40S | 4 | 421.7 | 0.509 | 1.251 | 1.251 | `final_logs/RESNET/RESNET34_gender_2s/resnet34_gender_2s.log` |
| resnet34 | 4s | 7,327,783 | NVIDIA L40S | 4 | 354.2 | 0.863 | 1.777 | 1.777 | `final_logs/RESNET/RESNET34_gender_4s/resnet34_gender_4s.log` |
| resnet50 | 1s | 16,296,504 | NVIDIA L40S | 4 | 755.7 | 0.431 | 1.154 | 1.154 | `final_logs/RESNET/RESNET50_gender_1s/resnet50_gender_1s.log` |
| resnet50 | 2s | 16,296,504 | NVIDIA L40S | 4 | 538.7 | 0.698 | 1.552 | 1.552 | `final_logs/RESNET/RESNET50_gender_2s/resnet50_gender_2s.log` |
| resnet50 | 4s | 16,296,504 | NVIDIA L40S | 4 | 277.8 | 0.665 | 1.406 | 1.406 | `final_logs/RESNET/RESNET50_gender_4s/resnet50_gender_4s.log` |

## Age Regression

### LaBraM

| Segment | Params | GPU | #GPUs | Test s/batch | Active s/batch | Passive s/batch | Source |
|---------|--------|-----|-------|--------------|----------------|-----------------|--------|
| 1s | 5,824,937 | NVIDIA L40S | 1 | 1.3094 | 0.0203 | 0.0161 | `final_logs/LaBraM/LABRAM_base_age_regression_1s/labram_base_age_regression_1s_resume2.log` |
| 2s | 5,824,937 | NVIDIA L40S | 2 | 0.1712 | 0.0262 | 0.0188 | `final_logs/LaBraM/LABRAM_base_age_regression_2s/labram_base_age_regression_2s.log` |
| 4s | 5,824,937 | NVIDIA L40S | 1 | 0.2511 | 0.0497 | 0.0468 | `final_logs/LaBraM/LABRAM_base_age_regression_4s/labram_base_age_regression_4s.log` |

### CNN

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| eeg_cnn | 1s | 593,522 | NVIDIA L40S | 4 | 929.2 | 0.542 | 1.407 | 1.407 | `final_logs/CNN/CNN_1s/cnn_1s.log` |
| eeg_cnn | 2s | 593,522 | NVIDIA L40S | 4 | 547.8 | 0.721 | 1.567 | 1.567 | `final_logs/CNN/CNN_2s/cnn_2s.log` |
| eeg_cnn | 4s | 593,522 | NVIDIA L40S | 4 | 342.0 | 0.823 | 1.727 | 1.727 | `final_logs/CNN/CNN_4s/cnn_4s.log` |

### ResNet

| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |
|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|
| resnet18 | 1s | 3,946,007 | NVIDIA L40S | 1 | 464.9 | 0.475 | 0.500 | 0.500 | `final_logs/RESNET/RESNET18_age_regression_1s/resnet18_age_regression_1s.log` |
| resnet18 | 2s | 3,946,007 | NVIDIA L40S | 1 | 276.8 | 0.552 | 0.604 | 0.604 | `final_logs/RESNET/RESNET18_age_regression_2s/resnet18_age_regression_2s.log` |
| resnet18 | 4s | 3,946,007 | NVIDIA L40S | 1 | 285.7 | 1.011 | 1.119 | 1.119 | `final_logs/RESNET/RESNET18_age_regression_4s/resnet18_age_regression_4s.log` |
| resnet34 | 1s | 7,327,783 | NVIDIA L40S | 1 | 546.7 | 0.474 | 0.672 | 0.672 | `final_logs/RESNET/RESNET34_age_regression_1s/resnet34_age_regression_1s.log` |
| resnet34 | 2s | 7,327,783 | NVIDIA L40S | 1 | 302.7 | 0.580 | 0.684 | 0.684 | `final_logs/RESNET/RESNET34_age_regression_2s/resnet34_age_regression_2s.log` |
| resnet34 | 4s | 7,327,783 | NVIDIA L40S | 1 | 292.8 | 1.047 | 1.135 | 1.135 | `final_logs/RESNET/RESNET34_age_regression_4s/resnet34_age_regression_4s.log` |
| resnet50 | 1s | 16,296,504 | NVIDIA L40S | 1 | 459.8 | 0.465 | 0.500 | 0.500 | `final_logs/RESNET/RESNET50_age_regression_1s/resnet50_age_regression_1s.log` |
| resnet50 | 2s | 16,296,504 | NVIDIA L40S | 1 | 282.3 | 0.557 | 0.622 | 0.622 | `final_logs/RESNET/RESNET50_age_regression_2s/resnet50_age_regression_2s.log` |
| resnet50 | 4s | 16,296,504 | NVIDIA L40S | 1 | 286.4 | 1.038 | 1.097 | 1.097 | `final_logs/RESNET/RESNET50_age_regression_4s/resnet50_age_regression_4s.log` |
