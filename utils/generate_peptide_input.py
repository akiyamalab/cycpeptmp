import pandas as pd
import numpy as np

from utils import calculate_descriptors



def generate_peptide_input(config, df, df_enu, df_pep_2D, df_pep_3D, folder_path, set_name):
    """
    Generate peptide input (peptide descriptors & Morgan fingerprint).
    """
    REPLICA_NUM = config['augmentation']['replica_num']
    use_descriptors = config['descriptor']['desc_2D'] + config['descriptor']['desc_3D']

    # Peptide information
    id = df_enu['ID'].to_numpy()
    smiles = df_enu['SMILES'].to_numpy()
    y = df['permeability'].to_numpy()
    y = np.clip(y, config['data']['lower_limit'], config['data']['upper_limit']).repeat(REPLICA_NUM)


    df_2D = df_pep_2D.iloc[sum([[_]*REPLICA_NUM for _ in range(len(df_pep_2D))], [])].reset_index(drop=True)
    df_2D = df_2D.iloc[:, df_2D.columns.to_list().index('apol'):].copy()
    df_3D = df_pep_3D.iloc[:, df_pep_3D.columns.to_list().index('ASA'):].copy()
    df_pep = pd.concat([df_2D, df_3D], axis=1)

    # Standardize peptide descriptors by Z-score
    desc_preprocessing = df_pep[use_descriptors].copy()
    pep_desc_mean = config['descriptor']['pep_desc_mean']
    pep_desc_std = config['descriptor']['pep_desc_std']

    for desc in desc_preprocessing:
        desc_preprocessing[desc] = (desc_preprocessing[desc] - pep_desc_mean[desc]) / pep_desc_std[desc]
    desc_preprocessing = desc_preprocessing.to_numpy()

    # Morgan fingerprint
    # NOTE: radius=2 and radius=3 are used
    fps_r2 = calculate_descriptors.calc_fingerprint(df['SMILES'].tolist(), 2, config['fingerprint']['bit_num'])
    fps_r3 = calculate_descriptors.calc_fingerprint(df['SMILES'].tolist(), 3, config['fingerprint']['bit_num'])
    fps = np.hstack([fps_r2, fps_r3])
    fps = fps.repeat(REPLICA_NUM)


    np.savez_compressed(f'{folder_path}/MLP/{REPLICA_NUM}/peptide_{REPLICA_NUM}_{set_name}.npz',
                        id=id,
                        smiles=smiles,
                        peptide_descriptor=desc_preprocessing,
                        fps=fps,
                        y=y)


    # # NOTE: You can also generate input data for different augmentation times
    # # For example:
    # total_list = list(range(0, len(df_enu)))
    # for replica_num in [1, 5, 10, 20, 30, 40, 50]:
    #     # IMPORTANT
    #     select_list = [total_list[i:i+replica_num] for i in range(0, len(total_list), REPLICA_NUM)]
    #     select_list = sum(select_list, [])
    #     np.savez_compressed(f'{folder_path}/MLP/{replica_num}/peptide_{replica_num}_{set_name}.npz',
    #                         id=id[select_list],
    #                         smiles=smiles[select_list],
    #                         peptide_descriptor=desc_preprocessing[select_list],
    #                         fps=fps[select_list],
    #                         y=y[select_list])