import pandas as pd
import numpy as np
import re
import os
from scipy.stats import pearsonr
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from utils import SmilesEnumerator





def canonicalize_smiles(smiles):
    """
    Return RDKit canonicalize SMILES.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))




def get_unique_monomer(df, file_path):
    """
    Get unique monomers from df and save it as a csv file.
    """

    if not os.path.exists(file_path):

        # Save unique monomers for descriptor calculation.
        monomer_list = df[df.filter(like='Substructure-').columns].values
        unique_monomer = np.unique(monomer_list[~pd.isna(monomer_list)])
        unique_monomer = [canonicalize_smiles(_) for _ in unique_monomer]
        unique_monomer_mw = [Chem.rdMolDescriptors._CalcMolWt(Chem.MolFromSmiles(_)) for _ in unique_monomer]
        df_monomer =  pd.DataFrame([unique_monomer, unique_monomer_mw], index=['SMILES', 'MolWt']).T
        # Sort by MolWt
        df_monomer = df_monomer.sort_values('MolWt').reset_index(drop=True)

        # Refer to 387 unique monomers of CycPeptMP experimental data to determine ID and Symbol.
        cycpeptmp_monomers = pd.read_csv('data/unique_monomer.csv')
        smiles_to_id = dict(zip(cycpeptmp_monomers['SMILES'], cycpeptmp_monomers['ID']))
        smiles_to_symbol = dict(zip(cycpeptmp_monomers['SMILES'], cycpeptmp_monomers['Symbol']))

        ID, Symbol = [], []
        cnt_id, cnt_symbol = 388, 92
        for smi in df_monomer['SMILES'].tolist():
            if smi in smiles_to_symbol:
                ID.append(smiles_to_id[smi])
                Symbol.append(smiles_to_symbol[smi])
            else:
                ID.append(cnt_id)
                Symbol.append(f'Sub{cnt_symbol}')
                cnt_id += 1
                cnt_symbol += 1

        df_monomer.insert(0, 'ID', ID)
        df_monomer.insert(1, 'Symbol', Symbol)

        df_monomer.to_csv(file_path, index=False)

    else:
        print('Unique monomer list already exists.')




def enumerate_smiles(df, config, file_path):
    """
    Enumerate REPLICA_NUM SMILES representations per peptide.
    """
    if not os.path.exists(file_path):

        REPLICA_NUM = config['augmentation']['replica_num']

        sme = SmilesEnumerator.SmilesEnumerator()

        id = df['ID'].tolist()
        smiles = df['SMILES'].tolist()
        # Canonical smiles
        smiles = [canonicalize_smiles(_) for _ in smiles]

        enu_id = [_ for _  in id for i in range(REPLICA_NUM)]
        enu_smi = []

        for i in tqdm(range(len(df))):
            for j in range(REPLICA_NUM):
                if j == 0:
                    enu_smi.append(smiles[i])
                else:
                    now_smi = sme.randomize_smiles(smiles[i])
                    count = 0
                    # NOTE: If a new SMILES is not generated after 1000 times, save the duplicated one.
                    while now_smi in enu_smi:
                        if count >= config['augmentation']['sme_dup_thresh']:
                            break
                        now_smi = sme.randomize_smiles(smiles[i])
                        count += 1
                    enu_smi.append(now_smi)

        df_enu = pd.DataFrame([enu_id, enu_smi], index=['ID', 'SMILES']).T
        df_enu.to_csv(file_path, index=False)

    else:
        print('Enumerated SMILES already exists.')




############################################################################################################




def combine_cxsmiles(cxsmiles, symbol, R3_dict):
    """
    Combine monomer of side chain to a substructure.
    """
    for i in range(len(cxsmiles)):
        tmp = cxsmiles[i].split(' |')[0]
        for _ in re.findall('_R\d', cxsmiles[i]):
            if _ == '_R1':
                tmp = tmp.replace('[*]', '[1C]', 1)
            elif _ == '_R2':
                tmp = tmp.replace('[*]', '[2C]', 1)
            elif _ == '_R3':
                # If R3 is not used, change it back to H or OH.
                tmp = tmp.replace('[*]', '['+R3_dict[symbol[i]]+']', 1)

        # Combine
        rxn = AllChem.ReactionFromSmarts('[*:1][2C].[1C][*:2]>>[*:1][*:2]')
        now_mol = Chem.MolFromSmiles(tmp)
        if i == 0:
            mol = now_mol
        else:
            mol = rxn.RunReactants((mol, now_mol))[0][0]

    # Replace unused R1 and R2.
    smi = canonicalize_smiles(Chem.MolToSmiles(mol).replace('[1C]', '[H]').replace('[2C]', '[OH]'))

    return smi




def delete_standard_deviation(x):
    """
    Delete features with standard deviation of 0.
    """
    features_delete_std = []
    for i in x:
        std = np.std(x[i])
        if (std == 0):
            features_delete_std.append(i)
    return features_delete_std




def delete_similarity(X, y, threshold=0.9):
    """
    For descriptor pairs whose correlation > th,
    eliminate the one with the lower correlation with the objective variable y.
    """
    features_delete = set()

    for i in range(X.shape[1]):
        if X.columns[i] in features_delete:
            continue
        a = X.iloc[:, i].values
        for j in range(i):
            if X.columns[j] in features_delete:
                continue
            b = X.iloc[:, j].values
            R = abs(pearsonr(a, b)[0])
            if R > threshold:
                cor_a = abs(pearsonr(a, y)[0])
                cor_b = abs(pearsonr(b, y)[0])
                if cor_a <= cor_b:
                    features_delete.add(X.columns[i])
                else:
                    features_delete.add(X.columns[j])

    return list(features_delete)



def entire_preprocessing(x, y, threshold=0.9):
    """
    Preprocess the peptide descriptors.
    """
    features_delete_std = delete_standard_deviation(x)
    x = x.drop(features_delete_std, axis=1)
    print(f'Deleted by standard deviation: {len(features_delete_std)}')

    features_delete_R = delete_similarity(x, y, threshold)
    x = x.drop(features_delete_R, axis=1)
    print(f'Deleted by similarity: {len(features_delete_R)}')

    # Z-score
    data_preprocessed = x.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    print(f'Feature map shape: {data_preprocessed.shape}')
    return features_delete_std, features_delete_R, data_preprocessed




def creat_internal_testset(x_variables, k):
    """""
    Creat testset by Kennard-Stone algorithm.
    𝑂(n_sample * n_result)
    """""

    x_variables = np.array(x_variables)
    original_x = x_variables
    # 1. Calculate the sample mean for the variables.
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(axis=1)
    # 2. Select the sample with the largest Euclidean distance to the mean.
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    # 6. Repeat steps 3 to 5.
    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        # 3. For each sample that has not yet been selected, calculate the Euclidean distance between it and all samples that have been selected.
        # 4. The minimum distance of 3 is taken as the representative distance of each sample.
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        # 5. Select the sample with the largest representative distance.
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

    return selected_sample_numbers, remaining_sample_numbers