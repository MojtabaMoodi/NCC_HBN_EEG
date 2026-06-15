# User identification (LaBraM ArcFace) — full report

- **Generated**: 2026-03-25
- **Source**: `final_user_identification`
- **Content**: cross-validation accuracies, top-10 channels (vanilla gradients), topography figures per fold.

## 1s segments (`.`)

*No `cv_summary.json` in this segment directory (per-fold tables below only).*

### fold_0

**Accuracies** (from `results.json`):

- validation: **96.57327785483125** %
- test: **96.56248540787897** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 17.467173 | 221.553500 |
| 2 | T7 | 16.192509 | 230.303390 |
| 3 | AF8 | 15.802638 | 171.698600 |
| 4 | T10 | 15.381573 | 151.964040 |
| 5 | FZ | 12.419060 | 195.732860 |
| 6 | F1 | 12.068488 | 102.955000 |
| 7 | CP1 | 10.961215 | 150.013660 |
| 8 | C1 | 10.905454 | 92.008330 |
| 9 | C6 | 10.781034 | 165.461430 |
| 10 | AF7 | 10.660781 | 90.154140 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | T10 | 12.942348 | 165.825710 |
| 2 | T7 | 12.348058 | 150.946550 |
| 3 | F1 | 10.175795 | 76.954510 |
| 4 | C2 | 9.324957 | 78.112870 |
| 5 | AF8 | 9.007800 | 91.037990 |
| 6 | CP1 | 8.568307 | 75.463684 |
| 7 | P2 | 7.727799 | 72.583540 |
| 8 | FZ | 7.547829 | 67.060300 |
| 9 | PZ | 7.019770 | 84.397675 |
| 10 | FC6 | 6.964512 | 69.885640 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 18.919620 | 238.131070 |
| 2 | AF8 | 17.015354 | 182.448550 |
| 3 | T7 | 16.879868 | 241.687160 |
| 4 | T10 | 15.819315 | 149.462900 |
| 5 | FZ | 13.288669 | 210.522580 |
| 6 | F1 | 12.407712 | 106.905540 |
| 7 | C1 | 11.674519 | 96.573600 |
| 8 | C6 | 11.611473 | 177.206860 |
| 9 | AF7 | 11.416431 | 94.765305 |
| 10 | CP1 | 11.388734 | 159.657800 |

#### Topography — fold_0

**combined**

![fold_0 combined](../../final_user_identification/fold_0/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_0 active](../../final_user_identification/fold_0/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_0 passive](../../final_user_identification/fold_0/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_1

**Accuracies** (from `results.json`):

- validation: **96.93568364368983** %
- test: **96.93828810369624** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 20.195770 | 393.507420 |
| 2 | AF8 | 17.848759 | 220.882050 |
| 3 | P6 | 16.126196 | 398.988980 |
| 4 | CP4 | 13.541840 | 319.259600 |
| 5 | FZ | 13.234622 | 487.383850 |
| 6 | CP1 | 12.971148 | 425.522460 |
| 7 | F9 | 12.424909 | 371.160220 |
| 8 | AF7 | 12.386635 | 249.740660 |
| 9 | T10 | 12.346292 | 346.610500 |
| 10 | F3 | 12.008059 | 308.277070 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 9.594585 | 118.774810 |
| 2 | C2 | 9.233595 | 223.870740 |
| 3 | FP1 | 8.936084 | 509.345280 |
| 4 | P6 | 8.647208 | 167.098560 |
| 5 | PO3 | 8.217928 | 398.780600 |
| 6 | F1 | 7.532939 | 162.003650 |
| 7 | CP4 | 7.334022 | 221.490130 |
| 8 | CP1 | 7.253155 | 88.366700 |
| 9 | F3 | 7.222483 | 258.266400 |
| 10 | T7 | 7.084365 | 189.875290 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 22.121391 | 416.064900 |
| 2 | AF8 | 19.299417 | 234.202200 |
| 3 | P6 | 17.440784 | 426.829900 |
| 4 | CP4 | 14.632651 | 333.373870 |
| 5 | FZ | 14.446870 | 523.873660 |
| 6 | CP1 | 13.975957 | 459.827760 |
| 7 | AF7 | 13.452179 | 265.764740 |
| 8 | T10 | 13.389537 | 370.850950 |
| 9 | F9 | 13.383247 | 397.866670 |
| 10 | OZ | 12.944239 | 668.875060 |

#### Topography — fold_1

**combined**

![fold_1 combined](../../final_user_identification/fold_1/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_1 active](../../final_user_identification/fold_1/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_1 passive](../../final_user_identification/fold_1/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_2

**Accuracies** (from `results.json`):

- validation: **96.99500936076329** %
- test: **97.02112811753003** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 17.728815 | 146.861480 |
| 2 | C2 | 17.152382 | 152.806520 |
| 3 | F9 | 14.427479 | 748.420040 |
| 4 | FP1 | 12.759159 | 647.333560 |
| 5 | FZ | 12.466610 | 288.639370 |
| 6 | C6 | 11.853016 | 133.611630 |
| 7 | AF7 | 11.530917 | 98.215100 |
| 8 | F3 | 11.525087 | 138.595780 |
| 9 | C1 | 10.948353 | 167.413650 |
| 10 | P6 | 10.922453 | 222.840870 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | F9 | 14.200628 | 1377.057900 |
| 2 | FZ | 11.505220 | 715.634460 |
| 3 | C2 | 11.149293 | 222.813220 |
| 4 | AF8 | 11.132091 | 163.470900 |
| 5 | FP1 | 10.736057 | 1218.919800 |
| 6 | F1 | 9.690297 | 300.191160 |
| 7 | CP4 | 9.244532 | 738.293460 |
| 8 | P5 | 9.124289 | 635.387270 |
| 9 | F8 | 9.062205 | 433.430900 |
| 10 | P6 | 8.800723 | 505.164860 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 18.893726 | 143.670580 |
| 2 | C2 | 18.212317 | 136.756150 |
| 3 | F9 | 14.470082 | 569.835800 |
| 4 | FP1 | 13.118692 | 480.721040 |
| 5 | FZ | 12.638386 | 88.469480 |
| 6 | C6 | 12.573309 | 120.646650 |
| 7 | F3 | 12.172641 | 128.833340 |
| 8 | AF7 | 12.165744 | 94.384590 |
| 9 | C1 | 11.493603 | 102.172170 |
| 10 | P6 | 11.297674 | 116.132385 |

#### Topography — fold_2

**combined**

![fold_2 combined](../../final_user_identification/fold_2/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_2 active](../../final_user_identification/fold_2/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_2 passive](../../final_user_identification/fold_2/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_3

**Accuracies** (from `results.json`):

- validation: **96.78429480125892** %
- test: **96.77573901790619** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 17.402150 | 153.923070 |
| 2 | C2 | 14.496681 | 112.696526 |
| 3 | T7 | 14.319109 | 479.333280 |
| 4 | F9 | 14.073503 | 403.458950 |
| 5 | T10 | 13.231372 | 254.458300 |
| 6 | FZ | 13.146322 | 195.392320 |
| 7 | C1 | 12.561352 | 97.899100 |
| 8 | AF7 | 11.566569 | 98.740944 |
| 9 | F1 | 11.491565 | 158.589110 |
| 10 | C6 | 10.951791 | 81.343710 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 10.348804 | 83.039520 |
| 2 | T7 | 9.234864 | 90.298096 |
| 3 | T10 | 8.664435 | 57.621357 |
| 4 | FZ | 8.491784 | 98.470430 |
| 5 | F1 | 8.473367 | 49.510525 |
| 6 | C2 | 8.300757 | 61.825848 |
| 7 | CP1 | 7.909405 | 76.038970 |
| 8 | F9 | 7.799529 | 67.768350 |
| 9 | C1 | 7.451881 | 66.979140 |
| 10 | PO7 | 6.921301 | 69.626910 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 18.641102 | 163.279250 |
| 2 | C2 | 15.585030 | 119.364190 |
| 3 | T7 | 15.212141 | 518.272400 |
| 4 | F9 | 15.175394 | 436.465270 |
| 5 | T10 | 14.034142 | 274.802580 |
| 6 | FZ | 13.964334 | 207.759610 |
| 7 | C1 | 13.459523 | 102.317450 |
| 8 | AF7 | 12.386228 | 104.142610 |
| 9 | F1 | 12.022573 | 170.660500 |
| 10 | C6 | 11.783248 | 85.566350 |

#### Topography — fold_3

**combined**

![fold_3 combined](../../final_user_identification/fold_3/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_3 active](../../final_user_identification/fold_3/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_3 passive](../../final_user_identification/fold_3/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_4

*No `results.json` (fold may be incomplete).*

#### Top channels — combined

*Missing `channel_ranking_vanilla_gradients.csv`.*

#### Top channels — active

*Missing `channel_ranking_vanilla_gradients.csv`.*

#### Top channels — passive

*Missing `channel_ranking_vanilla_gradients.csv`.*

#### Topography — fold_4

*combined: `topography_vanilla_gradients.png` not found.*

*active: `topography_vanilla_gradients.png` not found.*

*passive: `topography_vanilla_gradients.png` not found.*


---

## 2s segments (`2s`)

### Cross-validation summary

| metric | mean | std |
| --- | ---: | ---: |
| val accuracy (%) | 96.88939394526365 | 0.16454925160169392 |
| test accuracy (%) | 96.84624927355165 | 0.16724800846952098 |

**Per-fold (val / test %):**

- Fold 0: val=96.67884388663026, test=96.62434713971595
- Fold 1: val=96.93115444767163, test=96.88661973646889
- Fold 2: val=96.94699024077012, test=96.89027464781486
- Fold 3: val=96.78261561728887, test=96.75748253475045
- Fold 4: val=97.10736553395736, test=97.07252230900811

### fold_0

**Accuracies** (from `results.json`):

- validation: **96.67884388663026** %
- test: **96.62434713971595** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 7.377163 | 132.821200 |
| 2 | AF8 | 7.080581 | 124.141130 |
| 3 | AF7 | 5.263216 | 106.292730 |
| 4 | F3 | 4.718909 | 92.434525 |
| 5 | FZ | 4.606819 | 102.684210 |
| 6 | C6 | 4.360847 | 79.505875 |
| 7 | C1 | 4.308119 | 76.856320 |
| 8 | F2 | 4.261856 | 84.183830 |
| 9 | T10 | 4.253489 | 134.411470 |
| 10 | F1 | 4.157070 | 402.778200 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | F1 | 12.678806 | 1014.134000 |
| 2 | FC4 | 8.890092 | 692.567300 |
| 3 | F8 | 8.094143 | 579.334400 |
| 4 | T10 | 6.714020 | 308.942500 |
| 5 | PO7 | 6.578496 | 359.964540 |
| 6 | PO3 | 6.417819 | 477.418060 |
| 7 | F9 | 6.326538 | 352.732120 |
| 8 | CP5 | 6.223043 | 443.267100 |
| 9 | F10 | 6.131769 | 484.238040 |
| 10 | CP1 | 5.787158 | 334.474980 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 7.756231 | 122.882430 |
| 2 | AF8 | 7.442722 | 122.363495 |
| 3 | AF7 | 5.423390 | 98.033900 |
| 4 | F3 | 4.795396 | 78.531820 |
| 5 | FZ | 4.522430 | 67.426994 |
| 6 | C6 | 4.521574 | 71.929550 |
| 7 | C1 | 4.492060 | 69.839790 |
| 8 | F2 | 4.241015 | 61.835140 |
| 9 | CP4 | 4.190495 | 59.267765 |
| 10 | CP6 | 4.019666 | 60.776764 |

#### Topography — fold_0

**combined**

![fold_0 combined](../../final_user_identification/2s/fold_0/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_0 active](../../final_user_identification/2s/fold_0/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_0 passive](../../final_user_identification/2s/fold_0/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_1

**Accuracies** (from `results.json`):

- validation: **96.93115444767163** %
- test: **96.88661973646889** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | F9 | 8.825137 | 538.373660 |
| 2 | AF7 | 7.136694 | 186.678530 |
| 3 | AF8 | 7.020452 | 120.551370 |
| 4 | F3 | 6.922313 | 267.051500 |
| 5 | C2 | 6.862680 | 125.597470 |
| 6 | F8 | 6.231540 | 304.426570 |
| 7 | PO4 | 5.718290 | 164.117660 |
| 8 | P3 | 5.668520 | 200.520570 |
| 9 | F1 | 5.394840 | 290.939060 |
| 10 | C6 | 5.376428 | 102.122610 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 4.139166 | 95.421180 |
| 2 | C2 | 3.542762 | 75.508270 |
| 3 | AF7 | 3.485519 | 62.763900 |
| 4 | C6 | 2.952371 | 61.126244 |
| 5 | P3 | 2.883533 | 60.656673 |
| 6 | F9 | 2.815652 | 72.948240 |
| 7 | PO4 | 2.765722 | 62.221640 |
| 8 | F3 | 2.607472 | 48.834614 |
| 9 | P6 | 2.464467 | 47.530660 |
| 10 | FC3 | 2.274434 | 33.092040 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | F9 | 9.924850 | 584.721400 |
| 2 | AF7 | 7.805162 | 201.236970 |
| 3 | F3 | 7.712138 | 289.701080 |
| 4 | AF8 | 7.548387 | 124.592530 |
| 5 | C2 | 7.470597 | 132.726580 |
| 6 | F8 | 7.021280 | 330.899100 |
| 7 | PO4 | 6.259359 | 176.506840 |
| 8 | P3 | 6.179072 | 216.544020 |
| 9 | F1 | 6.067032 | 316.212950 |
| 10 | T7 | 5.876016 | 290.297970 |

#### Topography — fold_1

**combined**

![fold_1 combined](../../final_user_identification/2s/fold_1/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_1 active](../../final_user_identification/2s/fold_1/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_1 passive](../../final_user_identification/2s/fold_1/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_2

**Accuracies** (from `results.json`):

- validation: **96.94699024077012** %
- test: **96.89027464781486** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 6.978613 | 97.675900 |
| 2 | C2 | 6.266890 | 135.690750 |
| 3 | C1 | 5.640799 | 116.497820 |
| 4 | C6 | 5.609055 | 80.464760 |
| 5 | F3 | 5.411881 | 65.220710 |
| 6 | FC3 | 4.957230 | 70.234790 |
| 7 | FZ | 4.896547 | 91.658600 |
| 8 | F2 | 4.819928 | 63.337750 |
| 9 | AF7 | 4.735967 | 61.193363 |
| 10 | P6 | 4.536794 | 64.570790 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | F1 | 4.882432 | 159.474470 |
| 2 | FZ | 4.791749 | 117.657660 |
| 3 | AF8 | 4.223246 | 59.730050 |
| 4 | C1 | 4.132235 | 83.998055 |
| 5 | C6 | 3.973511 | 90.366290 |
| 6 | F3 | 3.923196 | 54.949590 |
| 7 | F8 | 3.893493 | 69.699850 |
| 8 | C2 | 3.741785 | 63.521120 |
| 9 | AF7 | 3.599796 | 57.338097 |
| 10 | CP4 | 3.581502 | 68.857590 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.485735 | 103.134840 |
| 2 | C2 | 6.732121 | 145.091980 |
| 3 | C1 | 5.918761 | 121.518140 |
| 4 | C6 | 5.910169 | 78.609856 |
| 5 | F3 | 5.685673 | 66.947860 |
| 6 | FC3 | 5.318108 | 74.721330 |
| 7 | F2 | 5.090173 | 65.000480 |
| 8 | AF7 | 4.945083 | 61.900517 |
| 9 | FZ | 4.915900 | 86.057720 |
| 10 | P6 | 4.771917 | 67.311830 |

#### Topography — fold_2

**combined**

![fold_2 combined](../../final_user_identification/2s/fold_2/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_2 active](../../final_user_identification/2s/fold_2/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_2 passive](../../final_user_identification/2s/fold_2/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_3

**Accuracies** (from `results.json`):

- validation: **96.78261561728887** %
- test: **96.75748253475045** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | FZ | 7.062874 | 62.881893 |
| 2 | AF8 | 7.037978 | 74.525660 |
| 3 | C2 | 6.942493 | 71.390740 |
| 4 | F3 | 6.283135 | 52.916565 |
| 5 | C6 | 4.919723 | 45.337470 |
| 6 | AF7 | 4.775954 | 40.888080 |
| 7 | FC3 | 4.774997 | 62.748848 |
| 8 | P6 | 4.612073 | 46.130940 |
| 9 | T10 | 4.568930 | 73.574570 |
| 10 | T7 | 4.433392 | 54.409600 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | FZ | 5.347111 | 60.118313 |
| 2 | F3 | 4.085412 | 37.992496 |
| 3 | AF8 | 3.996950 | 28.726349 |
| 4 | C2 | 3.873883 | 30.577145 |
| 5 | T7 | 3.549400 | 24.584673 |
| 6 | CP1 | 3.529354 | 24.146410 |
| 7 | F1 | 3.464352 | 17.565315 |
| 8 | T10 | 3.327732 | 21.489792 |
| 9 | PO8 | 3.263383 | 38.103962 |
| 10 | F8 | 3.180152 | 16.680866 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.594637 | 80.127410 |
| 2 | C2 | 7.504098 | 76.537480 |
| 3 | FZ | 7.376989 | 63.363710 |
| 4 | F3 | 6.685484 | 55.197000 |
| 5 | C6 | 5.269537 | 48.571503 |
| 6 | FC3 | 5.101917 | 67.499990 |
| 7 | AF7 | 5.094145 | 43.161255 |
| 8 | P6 | 4.888855 | 48.781190 |
| 9 | T10 | 4.796120 | 79.491860 |
| 10 | T7 | 4.595205 | 58.235374 |

#### Topography — fold_3

**combined**

![fold_3 combined](../../final_user_identification/2s/fold_3/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_3 active](../../final_user_identification/2s/fold_3/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_3 passive](../../final_user_identification/2s/fold_3/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_4

**Accuracies** (from `results.json`):

- validation: **97.10736553395736** %
- test: **97.07252230900811** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 9.553730 | 129.441100 |
| 2 | C2 | 8.189957 | 111.652374 |
| 3 | C6 | 6.009802 | 79.629210 |
| 4 | AF7 | 5.969280 | 80.578700 |
| 5 | FZ | 5.793390 | 97.081880 |
| 6 | F3 | 5.755105 | 77.552500 |
| 7 | F4 | 5.133785 | 69.737890 |
| 8 | FC3 | 4.988020 | 65.918720 |
| 9 | C1 | 4.874932 | 83.801600 |
| 10 | F7 | 4.840443 | 66.170360 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 4.943071 | 70.021240 |
| 2 | C2 | 4.436727 | 91.633545 |
| 3 | FZ | 3.537587 | 55.047905 |
| 4 | F3 | 3.414816 | 59.441063 |
| 5 | AF7 | 3.317998 | 49.735065 |
| 6 | C6 | 3.216511 | 52.965977 |
| 7 | P3 | 3.186056 | 69.650160 |
| 8 | FC3 | 3.029103 | 43.186170 |
| 9 | F9 | 2.917536 | 48.308395 |
| 10 | PO4 | 2.886416 | 46.986385 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 10.404537 | 137.577420 |
| 2 | C2 | 8.882894 | 115.021810 |
| 3 | C6 | 6.525782 | 83.598550 |
| 4 | AF7 | 6.459123 | 85.035866 |
| 5 | FZ | 6.210102 | 102.972740 |
| 6 | F3 | 6.187567 | 80.496060 |
| 7 | F4 | 5.573323 | 74.281670 |
| 8 | FC3 | 5.349547 | 69.285324 |
| 9 | C1 | 5.330168 | 89.986046 |
| 10 | F7 | 5.252750 | 70.368340 |

#### Topography — fold_4

**combined**

![fold_4 combined](../../final_user_identification/2s/fold_4/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_4 active](../../final_user_identification/2s/fold_4/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_4 passive](../../final_user_identification/2s/fold_4/saliency/passive/topography_vanilla_gradients.png){width=52%}


---

## 4s segments (`4s`)

### Cross-validation summary

| metric | mean | std |
| --- | ---: | ---: |
| val accuracy (%) | 96.89735277053441 | 0.2734811349705264 |
| test accuracy (%) | 96.68966187941018 | 0.3172375688758606 |

**Per-fold (val / test %):**

- Fold 0: val=96.60949756000868, test=96.42844345707324
- Fold 1: val=97.26391791753753, test=97.11148374533455
- Fold 2: val=97.01908317383312, test=96.83846015470408
- Fold 3: val=96.9504796572252, test=96.74278824524465
- Fold 4: val=96.64378554406751, test=96.32713379469435

### fold_0

**Accuracies** (from `results.json`):

- validation: **96.60949756000868** %
- test: **96.42844345707324** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 6.979796 | 99.432320 |
| 2 | C2 | 6.292717 | 127.947040 |
| 3 | T7 | 5.634703 | 105.155400 |
| 4 | F1 | 4.898872 | 124.028990 |
| 5 | FT7 | 4.749293 | 79.598335 |
| 6 | F9 | 4.647123 | 88.438900 |
| 7 | T10 | 4.601664 | 52.328712 |
| 8 | AF7 | 4.339877 | 50.592358 |
| 9 | P2 | 4.127163 | 54.027622 |
| 10 | F4 | 4.092402 | 82.989630 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | T7 | 8.673419 | 218.704240 |
| 2 | F1 | 8.486782 | 277.007660 |
| 3 | C2 | 8.062657 | 258.322020 |
| 4 | F9 | 7.142424 | 175.456970 |
| 5 | AF8 | 6.651731 | 96.497955 |
| 6 | FT7 | 6.291232 | 163.922320 |
| 7 | FC3 | 6.217725 | 229.535800 |
| 8 | T10 | 6.075960 | 98.481926 |
| 9 | CP1 | 6.038547 | 152.056690 |
| 10 | PO4 | 6.033335 | 229.470720 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.046995 | 100.034420 |
| 2 | C2 | 5.931796 | 78.239296 |
| 3 | T7 | 5.014703 | 59.804554 |
| 4 | FT7 | 4.434611 | 46.434723 |
| 5 | T10 | 4.300832 | 36.286050 |
| 6 | AF7 | 4.228237 | 39.529568 |
| 7 | F1 | 4.166672 | 53.664787 |
| 8 | F9 | 4.137869 | 56.150593 |
| 9 | C1 | 4.133401 | 46.031837 |
| 10 | FP2 | 3.932604 | 38.125076 |

#### Topography — fold_0

**combined**

![fold_0 combined](../../final_user_identification/4s/fold_0/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_0 active](../../final_user_identification/4s/fold_0/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_0 passive](../../final_user_identification/4s/fold_0/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_1

**Accuracies** (from `results.json`):

- validation: **97.26391791753753** %
- test: **97.11148374533455** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.213358 | 83.529076 |
| 2 | C2 | 5.512408 | 59.451760 |
| 3 | AF7 | 5.177181 | 82.534890 |
| 4 | F9 | 4.978859 | 333.847840 |
| 5 | F3 | 4.709735 | 57.647830 |
| 6 | C1 | 4.442932 | 55.368206 |
| 7 | T7 | 4.416421 | 97.254555 |
| 8 | P6 | 4.307962 | 61.153595 |
| 9 | C6 | 4.256014 | 55.002922 |
| 10 | FC3 | 4.075236 | 57.076126 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 5.149578 | 43.507317 |
| 2 | T7 | 4.277419 | 36.528206 |
| 3 | AF7 | 4.146155 | 34.323680 |
| 4 | F3 | 4.068406 | 45.412020 |
| 5 | F8 | 3.926734 | 26.639114 |
| 6 | C2 | 3.893083 | 33.161900 |
| 7 | F1 | 3.861569 | 17.374834 |
| 8 | C1 | 3.835438 | 39.751724 |
| 9 | T10 | 3.727348 | 26.560770 |
| 10 | P6 | 3.614776 | 26.419546 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.629191 | 89.443790 |
| 2 | C2 | 5.838846 | 63.428402 |
| 3 | AF7 | 5.385057 | 89.151540 |
| 4 | F9 | 5.256824 | 365.820980 |
| 5 | F3 | 4.839021 | 59.808075 |
| 6 | C1 | 4.565392 | 58.006557 |
| 7 | P6 | 4.447686 | 65.971370 |
| 8 | T7 | 4.444458 | 105.333060 |
| 9 | C6 | 4.395257 | 57.814877 |
| 10 | FC3 | 4.209946 | 61.545628 |

#### Topography — fold_1

**combined**

![fold_1 combined](../../final_user_identification/4s/fold_1/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_1 active](../../final_user_identification/4s/fold_1/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_1 passive](../../final_user_identification/4s/fold_1/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_2

**Accuracies** (from `results.json`):

- validation: **97.01908317383312** %
- test: **96.83846015470408** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 6.060161 | 50.170124 |
| 2 | AF8 | 5.311308 | 39.098747 |
| 3 | FZ | 4.846913 | 111.540310 |
| 4 | C6 | 4.465521 | 33.534840 |
| 5 | F8 | 4.465433 | 56.766020 |
| 6 | AF7 | 4.422334 | 37.069930 |
| 7 | F4 | 4.149500 | 42.252247 |
| 8 | C1 | 4.040419 | 34.955223 |
| 9 | F3 | 3.908669 | 37.625576 |
| 10 | T7 | 3.836232 | 65.733250 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 4.548093 | 46.088394 |
| 2 | FZ | 4.518361 | 77.773350 |
| 3 | F8 | 4.206010 | 25.638699 |
| 4 | T7 | 4.032319 | 52.940750 |
| 5 | AF8 | 3.968764 | 25.007706 |
| 6 | F1 | 3.899244 | 33.173916 |
| 7 | F9 | 3.571756 | 29.757956 |
| 8 | FC6 | 3.533915 | 51.063050 |
| 9 | C6 | 3.453322 | 25.505120 |
| 10 | F4 | 3.442631 | 24.283546 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | C2 | 6.366264 | 50.959816 |
| 2 | AF8 | 5.583140 | 41.370840 |
| 3 | FZ | 4.913429 | 117.192635 |
| 4 | C6 | 4.670371 | 34.939434 |
| 5 | AF7 | 4.627240 | 39.431010 |
| 6 | F8 | 4.517926 | 61.167620 |
| 7 | F4 | 4.292572 | 45.018547 |
| 8 | C1 | 4.229808 | 37.087124 |
| 9 | F3 | 4.022711 | 38.316560 |
| 10 | T7 | 3.796600 | 68.029940 |

#### Topography — fold_2

**combined**

![fold_2 combined](../../final_user_identification/4s/fold_2/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_2 active](../../final_user_identification/4s/fold_2/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_2 passive](../../final_user_identification/4s/fold_2/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_3

**Accuracies** (from `results.json`):

- validation: **96.9504796572252** %
- test: **96.74278824524465** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 6.729408 | 51.822960 |
| 2 | F3 | 6.463884 | 46.770084 |
| 3 | C2 | 6.124101 | 58.216263 |
| 4 | T7 | 5.505772 | 37.489662 |
| 5 | C6 | 4.925533 | 38.273636 |
| 6 | T10 | 4.729181 | 48.848625 |
| 7 | AF7 | 4.642321 | 36.180763 |
| 8 | F4 | 4.622309 | 36.789757 |
| 9 | C1 | 4.341275 | 37.176460 |
| 10 | FZ | 4.274381 | 34.270880 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | T7 | 5.694861 | 31.477652 |
| 2 | AF8 | 5.136236 | 44.198060 |
| 3 | F3 | 5.073360 | 44.192974 |
| 4 | C2 | 4.479621 | 32.675730 |
| 5 | T10 | 4.398169 | 31.030952 |
| 6 | F4 | 4.171724 | 38.436287 |
| 7 | F1 | 4.153676 | 31.212818 |
| 8 | CP1 | 4.016740 | 24.533155 |
| 9 | F9 | 3.963839 | 29.467049 |
| 10 | C6 | 3.827488 | 29.752832 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 7.050509 | 53.221024 |
| 2 | F3 | 6.744150 | 47.270718 |
| 3 | C2 | 6.455539 | 62.101080 |
| 4 | T7 | 5.467709 | 38.586937 |
| 5 | C6 | 5.146866 | 39.769897 |
| 6 | AF7 | 4.859498 | 37.506240 |
| 7 | T10 | 4.795877 | 51.701134 |
| 8 | F4 | 4.713093 | 36.456104 |
| 9 | C1 | 4.535027 | 38.624550 |
| 10 | FZ | 4.413099 | 35.145927 |

#### Topography — fold_3

**combined**

![fold_3 combined](../../final_user_identification/4s/fold_3/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_3 active](../../final_user_identification/4s/fold_3/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_3 passive](../../final_user_identification/4s/fold_3/saliency/passive/topography_vanilla_gradients.png){width=52%}

### fold_4

**Accuracies** (from `results.json`):

- validation: **96.64378554406751** %
- test: **96.32713379469435** %

#### Top channels — combined

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 8.259606 | 67.573880 |
| 2 | C2 | 5.707404 | 66.250305 |
| 3 | T7 | 5.649588 | 186.567580 |
| 4 | F4 | 5.535112 | 47.329850 |
| 5 | FC3 | 5.469425 | 62.962997 |
| 6 | P6 | 5.372756 | 44.055100 |
| 7 | FT7 | 4.476920 | 51.049328 |
| 8 | F9 | 4.458612 | 121.052050 |
| 9 | AF7 | 4.135534 | 36.258793 |
| 10 | CP1 | 4.003179 | 37.943980 |

#### Top channels — active

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | T7 | 9.732561 | 446.187040 |
| 2 | F9 | 6.887461 | 282.555080 |
| 3 | AF8 | 6.618648 | 60.195940 |
| 4 | F1 | 5.561778 | 206.055950 |
| 5 | F4 | 5.063005 | 66.202410 |
| 6 | FC3 | 5.001093 | 82.224630 |
| 7 | P6 | 4.672140 | 51.139282 |
| 8 | C2 | 4.577818 | 32.287792 |
| 9 | P2 | 4.434804 | 81.710310 |
| 10 | PO7 | 4.344345 | 108.948580 |

#### Top channels — passive

| rank | channel | mean importance | std |
| ---: | --- | ---: | ---: |
| 1 | AF8 | 8.591973 | 68.962860 |
| 2 | C2 | 5.936297 | 71.175800 |
| 3 | F4 | 5.630919 | 42.488407 |
| 4 | FC3 | 5.564453 | 58.280830 |
| 5 | P6 | 5.514791 | 42.469930 |
| 6 | T7 | 4.822891 | 39.401470 |
| 7 | FT7 | 4.531183 | 46.518345 |
| 8 | AF7 | 4.245813 | 36.959590 |
| 9 | P1 | 4.014974 | 38.651240 |
| 10 | CP1 | 3.969938 | 38.338333 |

#### Topography — fold_4

**combined**

![fold_4 combined](../../final_user_identification/4s/fold_4/saliency/topography_vanilla_gradients.png){width=52%}

**active**

![fold_4 active](../../final_user_identification/4s/fold_4/saliency/active/topography_vanilla_gradients.png){width=52%}

**passive**

![fold_4 passive](../../final_user_identification/4s/fold_4/saliency/passive/topography_vanilla_gradients.png){width=52%}


---

