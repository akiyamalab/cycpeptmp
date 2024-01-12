# CycPeptMP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## About
- The official Python implementation of the **CycPeptMP**.
- **CycPeptMP** is an accurate and efficient method for predicting the membrane permeability of cyclic peptides.
- We designed features for cyclic peptides at the atom, monomer, and peptide levels to concurrently capture both the local sequence variations and global conformational changes in cyclic peptides. We also applied data augmentation techniques at three scales to enhance model training efficiency.

  ![framework](https://github.com/akiyamalab/cycpeptmp/assets/44156441/c7bc4c2d-c195-4fb0-87aa-2676f0b2b6a0)



## Requirements
- Python: 3.9.6
- Numpy: 1.25.0
- Pandas: 1.4.4
- Pytorch: 2.0.0 (CUDA: 11.7)
- RDKit: 2022.09.5
- Mordred: 1.2.0
- *MOE: 2019.01　(commercial software)*




## Dataset
- The original cyclic peptide structure (SMILES) and experimentally determined membrane permeability (LogPexp) used in this study were all sourced from [**CycPeptMPDB**](http://cycpeptmpdb.com/).
  - Li J., Yanagisawa K., Sugita M., Fujie T., Ohue M., and Akiyama Y. [CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01573), _Journal of Chemical Information and Modeling_, **63**(7): 2240–2250, 2023.



## Code
- EXAMPLE.ipynb
  > Jupyter notebook with an example of prediction.

- EXAMPLE.py
  > Jupyter notebook with an example of prediction.



## Pretrained weights
- Weights of CycPeptMP (60 times augmentation) for three validation runs (_Fusion-60_cv*.cpt_).
- Weights of fusion model with no augmentation (_Fusion-1_cv*.cpt_) and 20 times augmentation (_Fusion-20_cv*.cpt_) for three validation runs in ablation studies.



## Reference
- Li J., Yanagisawa K., and Akiyama Y. CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation, _Briefings in Bioinformatics_, submitted.
- Li J., Yanagisawa K., and Akiyama Y. [CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation](https://www.biorxiv.org/content/10.1101/2023.12.25.573282v1), _bioRxiv preprint_, 2023, 2023.12. 25.573282.


## Contact
- Jianan Li: li@bi.c.titech.ac.jp
