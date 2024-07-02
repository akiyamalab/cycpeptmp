# CycPeptMP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## About open source
- The paper of CycPeptMP is currently under review, and reviewers may also submit comments for revision, so I have not fully published my code and data (the bioRxiv preview version of the paper also did not show the GitHub link).
- The peer review process is expected to be completed by the end of July.



## About
- Python implementation of **CycPeptMP**.
- **CycPeptMP** is an accurate and efficient method for predicting the membrane permeability of cyclic peptides.
- We designed features for cyclic peptides at the atom, monomer, and peptide levels to concurrently capture both the local sequence variations and global conformational changes in cyclic peptides. We also applied data augmentation techniques at three scales to enhance model training efficiency.

[framework.pdf](https://github.com/user-attachments/files/16062350/framework.pdf)



## Requirements
- Python: 3.9.6
- Numpy: 1.25.0
- Pandas: 1.4.4
- Pytorch: 2.0.0 (CUDA: 11.7)
- RDKit: 2022.09.5
- Mordred: 1.2.0
- *MOE: 2019.01　(commercial software)*




## Dataset
- The original cyclic peptide structure (SMILES) and experimentally determined membrane permeability (_LogPexp_) used in this study were all sourced from [**CycPeptMPDB**](http://cycpeptmpdb.com/).
  - Li J., Yanagisawa K., Sugita M., Fujie T., Ohue M., and Akiyama Y. [CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01573), _Journal of Chemical Information and Modeling_, **63**(7): 2240–2250, 2023.
- Selected PAMPA datasets used in this research are summarized in `all_data.csv`.


## Code
- `EXAMPLE.ipynb`
  > Jupyter notebook with an example of prediction.

- `atoms_model.py`
  > Transformer-based atom model using _Node_, _Bond_, _Graph_, and _Conf_ created from `atoms_input.py`.
  > The maximum number of heavy atoms in the input is 128.

- `monomers_model.py`
  > CNN-based monomer model using 16 monomer features created from `monomers_input.py`.
  > The maximum number of monomers in the input is 16.

- `peptides_model.py`
  > MLP-based peptide model using 16 peptide features and 2048-bit Morgan fingerprint.



## Pretrained weights
- Weights of CycPeptMP (60 times augmentation) for three validation runs (`Fusion-60_cv*.cpt`).
- Weights of fusion model with no augmentation (`Fusion-1_cv*.cpt`) and 20 times augmentation (`Fusion-20_cv*.cpt`) for three validation runs in ablation studies.



## Reference
- Li J., Yanagisawa K., and Akiyama Y. CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation, _Briefings in Bioinformatics_, under review.
- Li J., Yanagisawa K., and Akiyama Y. [CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation](https://www.biorxiv.org/content/10.1101/2023.12.25.573282v1), _bioRxiv preprint_, 2023, 2023.12. 25.573282.


## Contact
- Jianan Li: li@bi.c.titech.ac.jp
