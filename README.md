# CycPeptMP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



## About
- Python implementation of **CycPeptMP**.
- **CycPeptMP** is an accurate and efficient model for predicting the membrane permeability of cyclic peptides.
- We designed features for cyclic peptides at the atom, monomer, and peptide levels to concurrently capture both the local sequence variations and global conformational changes in cyclic peptides. We also applied data augmentation techniques at three scales to enhance model training efficiency.

![framework](https://github.com/akiyamalab/cycpeptmp/assets/44156441/cc57f68f-dc02-486d-beb6-d6e9f2bcb1ae)



## Requirements
- Python: 3.9.6
- Numpy: 1.25.0
- Pandas: 1.4.4
- Pytorch: 2.0.0 (CUDA: 11.7)
- RDKit: 2022.09.5
- Mordred: 1.2.0
- *MOE: 2019.01 (commercial software)*




## Dataset
- Original cyclic peptide structure (SMILES) and experimentally determined membrane permeability (_LogPexp_) used in this study (`data/CycPeptMPDB_Peptide_All.csv`) were all sourced from [**CycPeptMPDB**](http://cycpeptmpdb.com/).
  - Li J., Yanagisawa K., Sugita M., Fujie T., Ohue M., and Akiyama Y. [CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01573), _Journal of Chemical Information and Modeling_, **63**(7): 2240–2250, 2023.
- Correspondence table of peptides and their constituent monomers is summarized in `data/monomer_table.csv`.
- Data used in this experiment, with duplicates removed (all: 7,451->7,337, **PAMPA: 6,941->6,889**), is summarized in `desc/peptide_used.csv`.
- Dataset split index is stored in `data/eval_index/`.
  > `*_ID.npy` shows the CycPeptMPDB peptide ID, and `*_index.npy` shows the index in sorted `desc/peptide_used.csv`.




## Input files
- Complete input files for training can be downloaded from [Zenodo](https://zenodo.org/records/15166699).




## Code
- `Testset.ipynb`
  > Prediction for the test set (and other assay data) shown in the paper by CycPeptMP and other baselines.
  > Please download the input files `model/input/Trans/60/` from [Google Drive](https://drive.google.com/drive/folders/1BkkR2skuedOmiu87N6LMWHQhvBRyGklc?usp=sharing).

- `Newdata.ipynb`
  > Prediction for new data.

- `Train.ipynb`
  > Re-searching hyperparameters/training models.
  > Please refer to the process of `Newdata.ipynb` for the generation of input files.




## Pre-trained weights
- Weights of CycPeptMP (60 times augmentation) for three validation runs (`Fusion-60_cv*.cpt`).
- Weights of fusion model with no augmentation (`Fusion-1_cv*.cpt`) and 20 times augmentation (`Fusion-20_cv*.cpt`) for three validation runs in ablation studies.




## Reference
- Li J., Yanagisawa K., and Akiyama Y. [CycPeptMP: Enhancing Membrane Permeability Prediction of Cyclic Peptides with Multi-Level Molecular Features and Data Augmentation](https://doi.org/10.1093/bib/bbae417), _Briefings in Bioinformatics_, 2024, 25(5), bbae417.



## Contact
- Jianan Li: li@bi.c.titech.ac.jp
