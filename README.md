# ChemTS-torch

## Overview
ChemTS-torch is a PyTorch implementation based on previous ChemTS works, including ChemTSv2[^1]\([https://github.com/molecule-generator-collection/ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2)\), ChemTS[^2]\([https://github.com/tsudalab/ChemTS](https://github.com/tsudalab/ChemTS)\) and MPChemTS[^3]\([https://github.com/yoshizoe/mp-chemts](https://github.com/yoshizoe/mp-chemts)\).

**New features:**
- Generative model implemented by PyTorch;
- Multi-GPU support via PyTorch Lightning;
- Both SMILES[^4] and SELFIES[^5] available for string-based molecule generation.
- Transformer available as the generative model for better generation quality.

## Setup
```
cd <YOUR PATH>
git clone https://github.com/tsudalab/ChemTS-torch
cd ChemTS-torch
pip install -r requirements.txt
export PYTHONPATH=<YOUR PATH>/ChemTS-torch
```

## Train a RNN/Transformer generative model

Two pretrained models are provided in the `pretrained/` folder. They are trained on the 250k ZINC data with SMILES and SELFIES, respectively.

You can also follow the steps to train a custom RNN/Transformer molecule generative model:

1. Prepare a molecule data file in smiles format. If you want to train a SELFIES predictor, set it in the config file and it will be automatically converted during training.
2. Prepare a configuration file to set parameters for training the model.
3. Run the commands.
```
cd train_model
python train_RNN.py --config model_setting.yaml
```
4. The checkpoint file of the model with the highest validaiton accuracy will be saved to the path which is set in the configuration file. The checkpoint file is used to reload the trained model for the molecule generation process.

### Configuration file for training

Here is an example configuration file `model_setting.yaml`.

```
Data:
  dataset: ../data/250k_rndm_zinc_drugs_clean.smi               # path to the smiles file
  format: smiles                                                # string-based molecule representation: smiles or selfies
  output_model_dir: pretrained/smiles_zinc250k                  # directory to save model checkpoints
  output_token: pretrained/smiles_zinc250k/smiles_tokens.txt    # path to save tokens
  seq_len: 73                                                   # maximum length of the token sequences, automatically calculated
  vocab_len: 65                                                 # size of the token vocabulary, automatically calculated

Model:
  type: transformer                                             # which generative model to use: rnn or transformer
  dropout_rate: 0.1                                             # dropout rate
  hidden_dim: 512                                               # number of hidden features

  ### if rnn is used: ###
  n_layer: 2                                                    # number of recurrent layers
  ### if transformer is used: ###
  embed_dim: 128                                                # number of embedding features
  n_heads: 8                                                    # number of attention heads
  n_layer: 6                                                    # number of transformer blocks

Seed: 123                                                       # random seed

Train:
  accelerator: gpu                                              # cpu, gpu for training
  batch_size: 512                                               # batch size of data
  decay_alpha: 0.01                                             # decay rate of the optimizer scheduler
  decay_steps: 100                                              # decay steps of the optimizer scheduler
  device: 3                                                     # which gpu to use, for example, cuda:3
  epoch: 1000                                                   # training epochs
  gradient_clip: 2.0                                            # value for gradient clipping
  learning_rate: 0.001                                          # learning rate
  num_workers: 12                                               # number of workers for the dataloader
  optimizer: adam                                               # optimizer, adam or adamw
  patience: 50                                                  # patience for early stopping
  scheduler: CosineAnnealingLR                                  # optimizer scheduler
  validation_split: 0.1                                         # data split ratio for validation
```

## Molecule generation

```
python run.py --config config/setting.yaml
```
Details to be updated...

### Configuration file for de novo generation

To be updated...

## Reference
[^1]: Ishida, S., Aasawat, T., Sumita, M., Katouda, M., Yoshizawa, T., Yoshizoe, K., Tsuda, K. and Terayama, K., 2023. ChemTSv2: Functional molecular design using de novo molecule generator. Wiley Interdisciplinary Reviews: Computational Molecular Science, p.e1680.

[^2]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K. and Tsuda, K., 2017. ChemTS: an efficient python library for de novo molecular generation. Science and technology of advanced materials, 18(1), pp.972-976.

[^3]: Yang, X., Aasawat, T.K. and Yoshizoe, K., 2020. Practical massively parallel monte-carlo tree search applied to molecular design. arXiv preprint arXiv:2006.10504.

[^4]: Weininger, D., 1988. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1), pp.31-36.

[^5]: Krenn, M., Häse, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A., 2020. Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(4), p.045024.
