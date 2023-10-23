# ChemTS-torch

## Overview
ChemTS-torch is a PyTorch implementation based on previous ChemTS works, including ChemTSv2[^1]\([https://github.com/molecule-generator-collection/ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2)\), ChemTS[^2]\([https://github.com/tsudalab/ChemTS](https://github.com/tsudalab/ChemTS)\) and MPChemTS[^3]\([https://github.com/yoshizoe/mp-chemts](https://github.com/yoshizoe/mp-chemts)\).

**New features:**
- Generative model implemented by PyTorch;
- Multi-GPU support via PyTorch Lightning;
- Both SMILES and SELFIES[^4] available for string-based molecule generation.

## Setup
```
cd <YOUR PATH>
git clone https://github.com/tsudalab/ChemTS-torch
cd ChemTS-torch
pip install -r requirements.txt
export PYTHONPYTH=<YOUR PATH>/ChemTS-torch
```

## Molecule generation

To be updated

## Train new model

To be updated

## Reference
[^1]: Ishida, S., Aasawat, T., Sumita, M., Katouda, M., Yoshizawa, T., Yoshizoe, K., Tsuda, K. and Terayama, K., 2023. ChemTSv2: Functional molecular design using de novo molecule generator. Wiley Interdisciplinary Reviews: Computational Molecular Science, p.e1680.

[^2]: Yang, X., Zhang, J., Yoshizoe, K., Terayama, K. and Tsuda, K., 2017. ChemTS: an efficient python library for de novo molecular generation. Science and technology of advanced materials, 18(1), pp.972-976.

[^3]: Yang, X., Aasawat, T.K. and Yoshizoe, K., 2020. Practical massively parallel monte-carlo tree search applied to molecular design. arXiv preprint arXiv:2006.10504.

[^4]: Krenn, M., Häse, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A., 2020. Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(4), p.045024.