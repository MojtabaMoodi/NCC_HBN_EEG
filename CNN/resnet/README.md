# ResNet for EEG

ResNet models (ResNet18/34/50) for gender and age classification. All documentation lives in the CNN docs:

- **[../docs/ResNet.md](../docs/ResNet.md)** — architecture, usage, CLI options, ModelFactory, and file layout.

Scripts: `run_resnet_age.py` (age, 3 classes), `run_resnet_gender.py` (gender, 2 classes). Run from **CNN/** with `python -m resnet.run_resnet_age` or `python -m resnet.run_resnet_gender` (see docs/ResNet.md for options).
