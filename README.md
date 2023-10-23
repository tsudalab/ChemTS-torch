# ChemTS-torch

## Overview
ChemTS-torch is a PyTorch implementation based on previous ChemTS works, including ChemTSv2[^1]\([https://github.com/molecule-generator-collection/ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2)\), ChemTS[^2]\([https://github.com/tsudalab/ChemTS](https://github.com/tsudalab/ChemTS)\) and MPChemTS[^3]\([https://github.com/yoshizoe/mp-chemts](https://github.com/yoshizoe/mp-chemts)\).

**New features:**
- Generative model implemented by PyTorch;
- Multi-GPU support via PyTorch Lightning;
- Both SMILES[^4] and SELFIES[^5] available for string-based molecule generation.

## Setup
```
cd <YOUR PATH>
git clone https://github.com/tsudalab/ChemTS-torch
cd ChemTS-torch
pip install -r requirements.txt
export PYTHONPATH=<YOUR PATH>/ChemTS-torch
```

## Train a RNN generative model

Use following steps to train a custom RNN molecule generative model:

1. Prepare a molecule data file in smiles format.
2. Prepare a configuration file to set parameters for training the model.
3. Run the commands.
```
cd train_model
python train_RNN.py --config model_setting.yaml
```

### Configuration file for training

Here is an example configuration file `model_setting.yaml`.

```
Data:
  dataset: ../data/250k_rndm_zinc_drugs_clean.smi       # path to the smiles file
  format: selfies       # string-based molecule representation: smiles or selfies
  output_model_dir: pretrained/selfies_zinc250k         #
  output_token: pretrained/selfies_zinc250k/selfies_tokens.txt
  seq_len: 73
  vocab_len: 110
Model:
  dropout_rate: 0.2
  n_layer: 2
  hidden_dim: 256
Seed: 123
Train:
  accelerator: gpu
  batch_size: 512
  decay_alpha: 0.01
  decay_steps: 100
  device: 3
  epoch: 1000
  gradient_clip: 2.0
  learning_rate: 0.001
  num_workers: 12
  optimizer: adam
  patience: 50
  scheduler: CosineAnnealingLR
  validation_split: 0.1
```

## Molecule generation

### Configuration

To be updated

## Reference
[^1]: Ishida, S., Aasawat, T., Sumita, M., Katouda, M., Yoshizawa, T., Yoshizoe, K., Tsuda, K. and Terayama, K., 2023. ChemTSv2: Functional molecular design using de novo molecule generator. Wiley Interdisciplinary Reviews: Computational Molecular Science, p.e1680.

[^2]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K. and Tsuda, K., 2017. ChemTS: an efficient python library for de novo molecular generation. Science and technology of advanced materials, 18(1), pp.972-976.

[^3]: Yang, X., Aasawat, T.K. and Yoshizoe, K., 2020. Practical massively parallel monte-carlo tree search applied to molecular design. arXiv preprint arXiv:2006.10504.

[^4]: Weininger, D., 1988. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1), pp.31-36.

[^5]: Krenn, M., Häse, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A., 2020. Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(4), p.045024.