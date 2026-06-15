# final_saliency2 report

- **Generated**: 2026-03-25
- **Source**: `final_saliency2`
- **Experiments (runs)**: 33
- **Method**: vanilla gradients (see `topography_vanilla_gradients.png` per split)

## Index

| run | combined topo | active topo | passive topo | top-3 channels (combined) |
| --- | --- | --- | --- | --- |
| `age_cnn_1s_best` | yes | yes | yes | FC6 (0.25424507), FC4 (0.24757011), FT8 (0.18740457) |
| `age_cnn_2s_best` | yes | yes | yes | FC4 (0.1168676), FC6 (0.100280985), PZ (0.094916984) |
| `age_cnn_4s_best` | yes | yes | yes | FC4 (0.060349252), PZ (0.054993317), FT8 (0.048526663) |
| `age_cnn_reg_1s_best` | yes | yes | yes | FC6 (0.04225001), FC4 (0.041681517), PO4 (0.041544527) |
| `age_cnn_reg_2s_best` | yes | yes | yes | FC4 (0.029128052), PO4 (0.025379404), FC6 (0.024588374) |
| `age_cnn_reg_4s_best` | yes | yes | yes | FC4 (0.014987235), FC6 (0.010251991), F1 (0.00993351) |
| `age_labram_1s_best` | yes | yes | yes | FZ (0.45819166), PO7 (0.42643946), PO4 (0.4178875) |
| `age_labram_2s_best` | yes | yes | yes | PO4 (0.2033511), FZ (0.20296237), P5 (0.18552227) |
| `age_labram_4s_best` | yes | yes | yes | P6 (0.145037), P5 (0.14497593), PO4 (0.14478923) |
| `age_resnet18_1s_best` | yes | yes | yes | PZ (0.04511135), T10 (0.03828403), P4 (0.036595955) |
| `age_resnet18_2s_best` | yes | yes | yes | PZ (0.037246823), P4 (0.034450848), FC3 (0.033149045) |
| `age_resnet18_4s_best` | yes | yes | yes | PZ (0.01917952), P4 (0.017616833), AFZ (0.01756154) |
| `age_resnet34_1s_best` | yes | yes | yes | PZ (0.04256707), PO8 (0.041859757), CPZ (0.039214086) |
| `age_resnet34_2s_best` | yes | yes | yes | PO8 (0.032897048), PO3 (0.032778524), CPZ (0.032709487) |
| `age_resnet34_4s_best` | yes | yes | yes | P7 (0.025016867), P1 (0.022217143), CPZ (0.021678522) |
| `age_resnet50_1s_best` | yes | yes | yes | PO8 (0.055890813), PZ (0.055038765), P4 (0.05011395) |
| `age_resnet50_2s_best` | yes | yes | yes | PO8 (0.059632227), P4 (0.056785505), PZ (0.05605796) |
| `age_resnet50_4s_best` | yes | yes | yes | FC4 (0.031048702), PO4 (0.030454239), FC3 (0.0296644) |
| `gender_cnn_1s_best` | yes | yes | yes | C3 (0.38224503), FT8 (0.3516358), P2 (0.2551655) |
| `gender_cnn_2s_best` | yes | yes | yes | C3 (0.15605305), FT8 (0.13303383), P2 (0.10887482) |
| `gender_cnn_4s_best` | yes | yes | yes | FT8 (0.01846841), C3 (0.018087922), P2 (0.0149094565) |
| `gender_labram_1s_best` | yes | yes | yes | C3 (2.3537662), FZ (2.046666), FT7 (1.9456161) |
| `gender_labram_2s_best` | yes | yes | yes | FZ (0.89037883), C3 (0.83920395), PO7 (0.7695737) |
| `gender_labram_4s_best` | yes | yes | yes | C3 (0.025494745), PZ (0.024226353), FT8 (0.021104163) |
| `gender_resnet18_1s_best` | yes | yes | yes | C3 (0.079731226), FT8 (0.07829051), P2 (0.07761101) |
| `gender_resnet18_2s_best` | yes | yes | yes | C3 (0.08265433), FT8 (0.079676166), P2 (0.0767616) |
| `gender_resnet18_4s_best` | yes | yes | yes | FT8 (0.021044044), C3 (0.018764306), P2 (0.017883388) |
| `gender_resnet34_1s_best` | yes | yes | yes | FT8 (0.107887246), P2 (0.09766406), PZ (0.09722789) |
| `gender_resnet34_2s_best` | yes | yes | yes | PZ (0.094560534), C3 (0.0858943), FT8 (0.0808161) |
| `gender_resnet34_4s_best` | yes | yes | yes | FT8 (0.02339825), C3 (0.01918753), PZ (0.018796353) |
| `gender_resnet50_1s_best` | yes | yes | yes | FT8 (0.13107802), P2 (0.12056682), PZ (0.115381286) |
| `gender_resnet50_2s_best` | yes | yes | yes | FT8 (0.10578434), C3 (0.09947304), P2 (0.098584585) |
| `gender_resnet50_4s_best` | yes | yes | yes | FT8 (0.040515658), P2 (0.03504595), P8 (0.03426946) |

## Topography figures

For each run: **combined** (all data), **active**, **passive** task splits. Images are embedded from `final_saliency2/<run>/`.

### `age_cnn_1s_best`

#### Combined (train/eval pool)

![age_cnn_1s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_1s_best — Active tasks](../../final_saliency2/age_cnn_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_1s_best — Passive tasks](../../final_saliency2/age_cnn_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_cnn_2s_best`

#### Combined (train/eval pool)

![age_cnn_2s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_2s_best — Active tasks](../../final_saliency2/age_cnn_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_2s_best — Passive tasks](../../final_saliency2/age_cnn_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_cnn_4s_best`

#### Combined (train/eval pool)

![age_cnn_4s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_4s_best — Active tasks](../../final_saliency2/age_cnn_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_4s_best — Passive tasks](../../final_saliency2/age_cnn_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_cnn_reg_1s_best`

#### Combined (train/eval pool)

![age_cnn_reg_1s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_reg_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_reg_1s_best — Active tasks](../../final_saliency2/age_cnn_reg_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_reg_1s_best — Passive tasks](../../final_saliency2/age_cnn_reg_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_cnn_reg_2s_best`

#### Combined (train/eval pool)

![age_cnn_reg_2s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_reg_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_reg_2s_best — Active tasks](../../final_saliency2/age_cnn_reg_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_reg_2s_best — Passive tasks](../../final_saliency2/age_cnn_reg_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_cnn_reg_4s_best`

#### Combined (train/eval pool)

![age_cnn_reg_4s_best — Combined (train/eval pool)](../../final_saliency2/age_cnn_reg_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_cnn_reg_4s_best — Active tasks](../../final_saliency2/age_cnn_reg_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_cnn_reg_4s_best — Passive tasks](../../final_saliency2/age_cnn_reg_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_labram_1s_best`

#### Combined (train/eval pool)

![age_labram_1s_best — Combined (train/eval pool)](../../final_saliency2/age_labram_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_labram_1s_best — Active tasks](../../final_saliency2/age_labram_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_labram_1s_best — Passive tasks](../../final_saliency2/age_labram_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_labram_2s_best`

#### Combined (train/eval pool)

![age_labram_2s_best — Combined (train/eval pool)](../../final_saliency2/age_labram_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_labram_2s_best — Active tasks](../../final_saliency2/age_labram_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_labram_2s_best — Passive tasks](../../final_saliency2/age_labram_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_labram_4s_best`

#### Combined (train/eval pool)

![age_labram_4s_best — Combined (train/eval pool)](../../final_saliency2/age_labram_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_labram_4s_best — Active tasks](../../final_saliency2/age_labram_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_labram_4s_best — Passive tasks](../../final_saliency2/age_labram_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet18_1s_best`

#### Combined (train/eval pool)

![age_resnet18_1s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet18_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet18_1s_best — Active tasks](../../final_saliency2/age_resnet18_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet18_1s_best — Passive tasks](../../final_saliency2/age_resnet18_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet18_2s_best`

#### Combined (train/eval pool)

![age_resnet18_2s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet18_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet18_2s_best — Active tasks](../../final_saliency2/age_resnet18_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet18_2s_best — Passive tasks](../../final_saliency2/age_resnet18_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet18_4s_best`

#### Combined (train/eval pool)

![age_resnet18_4s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet18_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet18_4s_best — Active tasks](../../final_saliency2/age_resnet18_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet18_4s_best — Passive tasks](../../final_saliency2/age_resnet18_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet34_1s_best`

#### Combined (train/eval pool)

![age_resnet34_1s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet34_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet34_1s_best — Active tasks](../../final_saliency2/age_resnet34_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet34_1s_best — Passive tasks](../../final_saliency2/age_resnet34_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet34_2s_best`

#### Combined (train/eval pool)

![age_resnet34_2s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet34_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet34_2s_best — Active tasks](../../final_saliency2/age_resnet34_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet34_2s_best — Passive tasks](../../final_saliency2/age_resnet34_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet34_4s_best`

#### Combined (train/eval pool)

![age_resnet34_4s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet34_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet34_4s_best — Active tasks](../../final_saliency2/age_resnet34_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet34_4s_best — Passive tasks](../../final_saliency2/age_resnet34_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet50_1s_best`

#### Combined (train/eval pool)

![age_resnet50_1s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet50_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet50_1s_best — Active tasks](../../final_saliency2/age_resnet50_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet50_1s_best — Passive tasks](../../final_saliency2/age_resnet50_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet50_2s_best`

#### Combined (train/eval pool)

![age_resnet50_2s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet50_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet50_2s_best — Active tasks](../../final_saliency2/age_resnet50_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet50_2s_best — Passive tasks](../../final_saliency2/age_resnet50_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `age_resnet50_4s_best`

#### Combined (train/eval pool)

![age_resnet50_4s_best — Combined (train/eval pool)](../../final_saliency2/age_resnet50_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![age_resnet50_4s_best — Active tasks](../../final_saliency2/age_resnet50_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![age_resnet50_4s_best — Passive tasks](../../final_saliency2/age_resnet50_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_cnn_1s_best`

#### Combined (train/eval pool)

![gender_cnn_1s_best — Combined (train/eval pool)](../../final_saliency2/gender_cnn_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_cnn_1s_best — Active tasks](../../final_saliency2/gender_cnn_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_cnn_1s_best — Passive tasks](../../final_saliency2/gender_cnn_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_cnn_2s_best`

#### Combined (train/eval pool)

![gender_cnn_2s_best — Combined (train/eval pool)](../../final_saliency2/gender_cnn_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_cnn_2s_best — Active tasks](../../final_saliency2/gender_cnn_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_cnn_2s_best — Passive tasks](../../final_saliency2/gender_cnn_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_cnn_4s_best`

#### Combined (train/eval pool)

![gender_cnn_4s_best — Combined (train/eval pool)](../../final_saliency2/gender_cnn_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_cnn_4s_best — Active tasks](../../final_saliency2/gender_cnn_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_cnn_4s_best — Passive tasks](../../final_saliency2/gender_cnn_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_labram_1s_best`

#### Combined (train/eval pool)

![gender_labram_1s_best — Combined (train/eval pool)](../../final_saliency2/gender_labram_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_labram_1s_best — Active tasks](../../final_saliency2/gender_labram_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_labram_1s_best — Passive tasks](../../final_saliency2/gender_labram_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_labram_2s_best`

#### Combined (train/eval pool)

![gender_labram_2s_best — Combined (train/eval pool)](../../final_saliency2/gender_labram_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_labram_2s_best — Active tasks](../../final_saliency2/gender_labram_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_labram_2s_best — Passive tasks](../../final_saliency2/gender_labram_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_labram_4s_best`

#### Combined (train/eval pool)

![gender_labram_4s_best — Combined (train/eval pool)](../../final_saliency2/gender_labram_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_labram_4s_best — Active tasks](../../final_saliency2/gender_labram_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_labram_4s_best — Passive tasks](../../final_saliency2/gender_labram_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet18_1s_best`

#### Combined (train/eval pool)

![gender_resnet18_1s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet18_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet18_1s_best — Active tasks](../../final_saliency2/gender_resnet18_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet18_1s_best — Passive tasks](../../final_saliency2/gender_resnet18_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet18_2s_best`

#### Combined (train/eval pool)

![gender_resnet18_2s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet18_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet18_2s_best — Active tasks](../../final_saliency2/gender_resnet18_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet18_2s_best — Passive tasks](../../final_saliency2/gender_resnet18_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet18_4s_best`

#### Combined (train/eval pool)

![gender_resnet18_4s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet18_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet18_4s_best — Active tasks](../../final_saliency2/gender_resnet18_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet18_4s_best — Passive tasks](../../final_saliency2/gender_resnet18_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet34_1s_best`

#### Combined (train/eval pool)

![gender_resnet34_1s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet34_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet34_1s_best — Active tasks](../../final_saliency2/gender_resnet34_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet34_1s_best — Passive tasks](../../final_saliency2/gender_resnet34_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet34_2s_best`

#### Combined (train/eval pool)

![gender_resnet34_2s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet34_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet34_2s_best — Active tasks](../../final_saliency2/gender_resnet34_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet34_2s_best — Passive tasks](../../final_saliency2/gender_resnet34_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet34_4s_best`

#### Combined (train/eval pool)

![gender_resnet34_4s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet34_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet34_4s_best — Active tasks](../../final_saliency2/gender_resnet34_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet34_4s_best — Passive tasks](../../final_saliency2/gender_resnet34_4s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet50_1s_best`

#### Combined (train/eval pool)

![gender_resnet50_1s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet50_1s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet50_1s_best — Active tasks](../../final_saliency2/gender_resnet50_1s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet50_1s_best — Passive tasks](../../final_saliency2/gender_resnet50_1s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet50_2s_best`

#### Combined (train/eval pool)

![gender_resnet50_2s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet50_2s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet50_2s_best — Active tasks](../../final_saliency2/gender_resnet50_2s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet50_2s_best — Passive tasks](../../final_saliency2/gender_resnet50_2s_best/passive/topography_vanilla_gradients.png){width=58%}

### `gender_resnet50_4s_best`

#### Combined (train/eval pool)

![gender_resnet50_4s_best — Combined (train/eval pool)](../../final_saliency2/gender_resnet50_4s_best/topography_vanilla_gradients.png){width=58%}

#### Active tasks

![gender_resnet50_4s_best — Active tasks](../../final_saliency2/gender_resnet50_4s_best/active/topography_vanilla_gradients.png){width=58%}

#### Passive tasks

![gender_resnet50_4s_best — Passive tasks](../../final_saliency2/gender_resnet50_4s_best/passive/topography_vanilla_gradients.png){width=58%}

## File inventory

Per-run counts of generated artifacts (full path list: `final_saliency2_index.json`).

| run | PNG files | CSV files |
| --- | ---: | ---: |
| `age_cnn_1s_best` | 21 | 3 |
| `age_cnn_2s_best` | 21 | 3 |
| `age_cnn_4s_best` | 21 | 3 |
| `age_cnn_reg_1s_best` | 21 | 3 |
| `age_cnn_reg_2s_best` | 21 | 3 |
| `age_cnn_reg_4s_best` | 21 | 3 |
| `age_labram_1s_best` | 21 | 3 |
| `age_labram_2s_best` | 21 | 3 |
| `age_labram_4s_best` | 21 | 3 |
| `age_resnet18_1s_best` | 21 | 3 |
| `age_resnet18_2s_best` | 21 | 3 |
| `age_resnet18_4s_best` | 21 | 3 |
| `age_resnet34_1s_best` | 19 | 3 |
| `age_resnet34_2s_best` | 21 | 3 |
| `age_resnet34_4s_best` | 21 | 3 |
| `age_resnet50_1s_best` | 21 | 3 |
| `age_resnet50_2s_best` | 21 | 3 |
| `age_resnet50_4s_best` | 21 | 3 |
| `gender_cnn_1s_best` | 21 | 3 |
| `gender_cnn_2s_best` | 21 | 3 |
| `gender_cnn_4s_best` | 21 | 3 |
| `gender_labram_1s_best` | 21 | 3 |
| `gender_labram_2s_best` | 21 | 3 |
| `gender_labram_4s_best` | 21 | 3 |
| `gender_resnet18_1s_best` | 21 | 3 |
| `gender_resnet18_2s_best` | 21 | 3 |
| `gender_resnet18_4s_best` | 21 | 3 |
| `gender_resnet34_1s_best` | 21 | 3 |
| `gender_resnet34_2s_best` | 21 | 3 |
| `gender_resnet34_4s_best` | 21 | 3 |
| `gender_resnet50_1s_best` | 21 | 3 |
| `gender_resnet50_2s_best` | 21 | 3 |
| `gender_resnet50_4s_best` | 21 | 3 |

