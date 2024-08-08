import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors




def canonicalize_smiles(smiles):
    """
    Return RDKit canonicalize SMILES.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))



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



def calc_rdkit_descriptors(smiles_list, mol_type):
    """
    Calculate 208 types RDKit 2D descriptors.
    """
    descs = [desc_name[0] for desc_name in Chem.Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    # Add Hs
    df_desc = pd.DataFrame([desc_calc.CalcDescriptors(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in tqdm(smiles_list)])
    df_desc.columns = descs
    df_desc.to_csv(f'desc/{mol_type}_rdkit.csv', index=False)



def calc_mordred_2Ddescriptors(smiles_list, mol_type):
    """
    Calculate 1275 types (correctly calculated) Mordred 2D descriptors.
    Some calculations may not be possible if the computing environment or input molecules are different.
    """
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    my_desc_names = ['ABC','ABCGG','nAcid','nBase','SpAbs_A','SpMax_A','SpDiam_A','SpAD_A','SpMAD_A','LogEE_A','VE1_A','VE2_A','VE3_A','VR1_A','VR2_A','VR3_A','nAromAtom','nAromBond','nAtom','nHeavyAtom','nSpiro','nBridgehead','nHetero','nH','nB','nC','nN','nO','nS','nP','nF','nCl','nBr','nI','nX','ATS0dv','ATS1dv','ATS2dv','ATS3dv','ATS4dv','ATS5dv','ATS6dv','ATS7dv','ATS8dv','ATS0d','ATS1d','ATS2d','ATS3d','ATS4d','ATS5d','ATS6d','ATS7d','ATS8d','ATS0s','ATS1s','ATS2s','ATS3s','ATS4s','ATS5s','ATS6s','ATS7s','ATS8s','ATS0Z','ATS1Z','ATS2Z','ATS3Z','ATS4Z','ATS5Z','ATS6Z','ATS7Z','ATS8Z','ATS0m','ATS1m','ATS2m','ATS3m','ATS4m','ATS5m','ATS6m','ATS7m','ATS8m','ATS0v','ATS1v','ATS2v','ATS3v','ATS4v','ATS5v','ATS6v','ATS7v','ATS8v','ATS0se','ATS1se','ATS2se','ATS3se','ATS4se','ATS5se','ATS6se','ATS7se','ATS8se','ATS0pe','ATS1pe','ATS2pe','ATS3pe','ATS4pe','ATS5pe','ATS6pe','ATS7pe','ATS8pe','ATS0are','ATS1are','ATS2are','ATS3are','ATS4are','ATS5are','ATS6are','ATS7are','ATS8are','ATS0p','ATS1p','ATS2p','ATS3p','ATS4p','ATS5p','ATS6p','ATS7p','ATS8p','ATS0i','ATS1i','ATS2i','ATS3i','ATS4i','ATS5i','ATS6i','ATS7i','ATS8i','AATS0dv','AATS1dv','AATS2dv','AATS3dv','AATS4dv','AATS5dv','AATS0d','AATS1d','AATS2d','AATS3d','AATS4d','AATS5d','AATS0s','AATS1s','AATS2s','AATS3s','AATS4s','AATS5s','AATS0Z','AATS1Z','AATS2Z','AATS3Z','AATS4Z','AATS5Z','AATS0m','AATS1m','AATS2m','AATS3m','AATS4m','AATS5m','AATS0v','AATS1v','AATS2v','AATS3v','AATS4v','AATS5v','AATS0se','AATS1se','AATS2se','AATS3se','AATS4se','AATS5se','AATS0pe','AATS1pe','AATS2pe','AATS3pe','AATS4pe','AATS5pe','AATS0are','AATS1are','AATS2are','AATS3are','AATS4are','AATS5are','AATS0p','AATS1p','AATS2p','AATS3p','AATS4p','AATS5p','AATS0i','AATS1i','AATS2i','AATS3i','AATS4i','AATS5i','ATSC0c','ATSC1c','ATSC2c','ATSC3c','ATSC4c','ATSC5c','ATSC6c','ATSC7c','ATSC8c','ATSC0dv','ATSC1dv','ATSC2dv','ATSC3dv','ATSC4dv','ATSC5dv','ATSC6dv','ATSC7dv','ATSC8dv','ATSC0d','ATSC1d','ATSC2d','ATSC3d','ATSC4d','ATSC5d','ATSC6d','ATSC7d','ATSC8d','ATSC0s','ATSC1s','ATSC2s','ATSC3s','ATSC4s','ATSC5s','ATSC6s','ATSC7s','ATSC8s','ATSC0Z','ATSC1Z','ATSC2Z','ATSC3Z','ATSC4Z','ATSC5Z','ATSC6Z','ATSC7Z','ATSC8Z','ATSC0m','ATSC1m','ATSC2m','ATSC3m','ATSC4m','ATSC5m','ATSC6m','ATSC7m','ATSC8m','ATSC0v','ATSC1v','ATSC2v','ATSC3v','ATSC4v','ATSC5v','ATSC6v','ATSC7v','ATSC8v','ATSC0se','ATSC1se','ATSC2se','ATSC3se','ATSC4se','ATSC5se','ATSC6se','ATSC7se','ATSC8se','ATSC0pe','ATSC1pe','ATSC2pe','ATSC3pe','ATSC4pe','ATSC5pe','ATSC6pe','ATSC7pe','ATSC8pe','ATSC0are','ATSC1are','ATSC2are','ATSC3are','ATSC4are','ATSC5are','ATSC6are','ATSC7are','ATSC8are','ATSC0p','ATSC1p','ATSC2p','ATSC3p','ATSC4p','ATSC5p','ATSC6p','ATSC7p','ATSC8p','ATSC0i','ATSC1i','ATSC2i','ATSC3i','ATSC4i','ATSC5i','ATSC6i','ATSC7i','ATSC8i','AATSC0c','AATSC1c','AATSC2c','AATSC3c','AATSC4c','AATSC5c','AATSC0dv','AATSC1dv','AATSC2dv','AATSC3dv','AATSC4dv','AATSC5dv','AATSC0d','AATSC1d','AATSC2d','AATSC3d','AATSC4d','AATSC5d','AATSC0s','AATSC1s','AATSC2s','AATSC3s','AATSC4s','AATSC5s','AATSC0Z','AATSC1Z','AATSC2Z','AATSC3Z','AATSC4Z','AATSC5Z','AATSC0m','AATSC1m','AATSC2m','AATSC3m','AATSC4m','AATSC5m','AATSC0v','AATSC1v','AATSC2v','AATSC3v','AATSC4v','AATSC5v','AATSC0se','AATSC1se','AATSC2se','AATSC3se','AATSC4se','AATSC5se','AATSC0pe','AATSC1pe','AATSC2pe','AATSC3pe','AATSC4pe','AATSC5pe','AATSC0are','AATSC1are','AATSC2are','AATSC3are','AATSC4are','AATSC5are','AATSC0p','AATSC1p','AATSC2p','AATSC3p','AATSC4p','AATSC5p','AATSC0i','AATSC1i','AATSC2i','AATSC3i','AATSC4i','AATSC5i','MATS1c','MATS2c','MATS3c','MATS4c','MATS5c','MATS1dv','MATS2dv','MATS3dv','MATS4dv','MATS5dv','MATS1d','MATS2d','MATS3d','MATS4d','MATS5d','MATS1s','MATS2s','MATS3s','MATS4s','MATS5s','MATS1Z','MATS2Z','MATS3Z','MATS4Z','MATS5Z','MATS1m','MATS2m','MATS3m','MATS4m','MATS5m','MATS1v','MATS2v','MATS3v','MATS4v','MATS5v','MATS1se','MATS2se','MATS3se','MATS4se','MATS5se','MATS1pe','MATS2pe','MATS3pe','MATS4pe','MATS5pe','MATS1are','MATS2are','MATS3are','MATS4are','MATS5are','MATS1p','MATS2p','MATS3p','MATS4p','MATS5p','MATS1i','MATS2i','MATS3i','MATS4i','MATS5i','GATS1c','GATS2c','GATS3c','GATS4c','GATS5c','GATS1dv','GATS2dv','GATS3dv','GATS4dv','GATS5dv','GATS1d','GATS2d','GATS3d','GATS4d','GATS5d','GATS1s','GATS2s','GATS3s','GATS4s','GATS5s','GATS1Z','GATS2Z','GATS3Z','GATS4Z','GATS5Z','GATS1m','GATS2m','GATS3m','GATS4m','GATS5m','GATS1v','GATS2v','GATS3v','GATS4v','GATS5v','GATS1se','GATS2se','GATS3se','GATS4se','GATS5se','GATS1pe','GATS2pe','GATS3pe','GATS4pe','GATS5pe','GATS1are','GATS2are','GATS3are','GATS4are','GATS5are','GATS1p','GATS2p','GATS3p','GATS4p','GATS5p','GATS1i','GATS2i','GATS3i','GATS4i','GATS5i','BCUTc-1h','BCUTc-1l','BCUTdv-1h','BCUTdv-1l','BCUTd-1h','BCUTd-1l','BCUTs-1h','BCUTs-1l','BCUTZ-1h','BCUTZ-1l','BCUTm-1h','BCUTm-1l','BCUTv-1h','BCUTv-1l','BCUTse-1h','BCUTse-1l','BCUTpe-1h','BCUTpe-1l','BCUTare-1h','BCUTare-1l','BCUTp-1h','BCUTp-1l','BCUTi-1h','BCUTi-1l','BalabanJ','SpAbs_DzZ','SpMax_DzZ','SpDiam_DzZ','SpAD_DzZ','SpMAD_DzZ','LogEE_DzZ','SM1_DzZ','VE1_DzZ','VE2_DzZ','VE3_DzZ','VR1_DzZ','VR2_DzZ','VR3_DzZ','SpAbs_Dzm','SpMax_Dzm','SpDiam_Dzm','SpAD_Dzm','SpMAD_Dzm','LogEE_Dzm','SM1_Dzm','VE1_Dzm','VE2_Dzm','VE3_Dzm','VR1_Dzm','VR2_Dzm','VR3_Dzm','SpAbs_Dzv','SpMax_Dzv','SpDiam_Dzv','SpAD_Dzv','SpMAD_Dzv','LogEE_Dzv','SM1_Dzv','VE1_Dzv','VE2_Dzv','VE3_Dzv','VR1_Dzv','VR2_Dzv','VR3_Dzv','SpAbs_Dzse','SpMax_Dzse','SpDiam_Dzse','SpAD_Dzse','SpMAD_Dzse','LogEE_Dzse','SM1_Dzse','VE1_Dzse','VE2_Dzse','VE3_Dzse','VR1_Dzse','VR2_Dzse','VR3_Dzse','SpAbs_Dzpe','SpMax_Dzpe','SpDiam_Dzpe','SpAD_Dzpe','SpMAD_Dzpe','LogEE_Dzpe','SM1_Dzpe','VE1_Dzpe','VE2_Dzpe','VE3_Dzpe','VR1_Dzpe','VR2_Dzpe','VR3_Dzpe','SpAbs_Dzare','SpMax_Dzare','SpDiam_Dzare','SpAD_Dzare','SpMAD_Dzare','LogEE_Dzare','SM1_Dzare','VE1_Dzare','VE2_Dzare','VE3_Dzare','VR1_Dzare','VR2_Dzare','VR3_Dzare','SpAbs_Dzp','SpMax_Dzp','SpDiam_Dzp','SpAD_Dzp','SpMAD_Dzp','LogEE_Dzp','SM1_Dzp','VE1_Dzp','VE2_Dzp','VE3_Dzp','VR1_Dzp','VR2_Dzp','VR3_Dzp','SpAbs_Dzi','SpMax_Dzi','SpDiam_Dzi','SpAD_Dzi','SpMAD_Dzi','LogEE_Dzi','SM1_Dzi','VE1_Dzi','VE2_Dzi','VE3_Dzi','VR1_Dzi','VR2_Dzi','VR3_Dzi','BertzCT','nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM','nBondsKS','nBondsKD','RNCG','RPCG','C1SP1','C2SP1','C1SP2','C2SP2','C3SP2','C1SP3','C2SP3','C3SP3','C4SP3','HybRatio','FCSP3','Xch-3d','Xch-4d','Xch-5d','Xch-6d','Xch-7d','Xch-3dv','Xch-4dv','Xch-5dv','Xch-6dv','Xch-7dv','Xc-3d','Xc-4d','Xc-5d','Xc-6d','Xc-3dv','Xc-4dv','Xc-5dv','Xc-6dv','Xpc-4d','Xpc-5d','Xpc-6d','Xpc-4dv','Xpc-5dv','Xpc-6dv','Xp-0d','Xp-1d','Xp-2d','Xp-3d','Xp-4d','Xp-5d','Xp-6d','Xp-7d','AXp-0d','AXp-1d','AXp-2d','AXp-3d','AXp-4d','Xp-0dv','Xp-1dv','Xp-2dv','Xp-3dv','Xp-4dv','Xp-5dv','Xp-6dv','Xp-7dv','AXp-0dv','AXp-1dv','AXp-2dv','AXp-3dv','AXp-4dv','SZ','Sm','Sv','Sse','Spe','Sare','Sp','Si','MZ','Mm','Mv','Mse','Mpe','Mare','Mp','Mi','SpAbs_D','SpMax_D','SpDiam_D','SpAD_D','SpMAD_D','LogEE_D','VE1_D','VE2_D','VE3_D','VR1_D','VR2_D','VR3_D','NsLi','NssBe','NssssBe','NssBH','NsssB','NssssB','NsCH3','NdCH2','NssCH2','NtCH','NdsCH','NaaCH','NsssCH','NddC','NtsC','NdssC','NaasC','NaaaC','NssssC','NsNH3','NsNH2','NssNH2','NdNH','NssNH','NaaNH','NtN','NsssNH','NdsN','NaaN','NsssN','NddsN','NaasN','NssssN','NsOH','NdO','NssO','NaaO','NsF','NsSiH3','NssSiH2','NsssSiH','NssssSi','NsPH2','NssPH','NsssP','NdsssP','NsssssP','NsSH','NdS','NssS','NaaS','NdssS','NddssS','NsCl','NsGeH3','NssGeH2','NsssGeH','NssssGe','NsAsH2','NssAsH','NsssAs','NsssdAs','NsssssAs','NsSeH','NdSe','NssSe','NaaSe','NdssSe','NddssSe','NsBr','NsSnH3','NssSnH2','NsssSnH','NssssSn','NsI','NsPbH3','NssPbH2','NsssPbH','NssssPb','SsLi','SssBe','SssssBe','SssBH','SsssB','SssssB','SsCH3','SdCH2','SssCH2','StCH','SdsCH','SaaCH','SsssCH','SddC','StsC','SdssC','SaasC','SaaaC','SssssC','SsNH3','SsNH2','SssNH2','SdNH','SssNH','SaaNH','StN','SsssNH','SdsN','SaaN','SsssN','SddsN','SaasN','SssssN','SsOH','SdO','SssO','SaaO','SsF','SsSiH3','SssSiH2','SsssSiH','SssssSi','SsPH2','SssPH','SsssP','SdsssP','SsssssP','SsSH','SdS','SssS','SaaS','SdssS','SddssS','SsCl','SsGeH3','SssGeH2','SsssGeH','SssssGe','SsAsH2','SssAsH','SsssAs','SsssdAs','SsssssAs','SsSeH','SdSe','SssSe','SaaSe','SdssSe','SddssSe','SsBr','SsSnH3','SssSnH2','SsssSnH','SssssSn','SsI','SsPbH3','SssPbH2','SsssPbH','SssssPb','MAXdO','MINdO','ECIndex','ETA_alpha','AETA_alpha','ETA_shape_p','ETA_shape_y','ETA_shape_x','ETA_beta','AETA_beta','ETA_beta_s','AETA_beta_s','ETA_beta_ns','AETA_beta_ns','ETA_beta_ns_d','AETA_beta_ns_d','ETA_eta','AETA_eta','ETA_eta_L','AETA_eta_L','ETA_eta_R','AETA_eta_R','ETA_eta_RL','AETA_eta_RL','ETA_eta_F','AETA_eta_F','ETA_eta_FL','AETA_eta_FL','ETA_eta_B','AETA_eta_B','ETA_eta_BR','AETA_eta_BR','ETA_dAlpha_A','ETA_dAlpha_B','ETA_epsilon_1','ETA_epsilon_2','ETA_epsilon_3','ETA_epsilon_4','ETA_epsilon_5','ETA_dEpsilon_A','ETA_dEpsilon_B','ETA_dEpsilon_C','ETA_dEpsilon_D','ETA_dBeta','AETA_dBeta','ETA_psi_1','ETA_dPsi_A','ETA_dPsi_B','fragCpx','fMF','nHBAcc','nHBDon','IC0','IC1','IC2','IC3','IC4','IC5','TIC0','TIC1','TIC2','TIC3','TIC4','TIC5','SIC0','SIC1','SIC2','SIC3','SIC4','SIC5','BIC0','BIC1','BIC2','BIC3','BIC4','BIC5','CIC0','CIC1','CIC2','CIC3','CIC4','CIC5','MIC0','MIC1','MIC2','MIC3','MIC4','MIC5','ZMIC0','ZMIC1','ZMIC2','ZMIC3','ZMIC4','ZMIC5','Kier1','Kier2','Kier3','FilterItLogS','VMcGowan','LabuteASA','PEOE_VSA1','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','SMR_VSA1','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','SlogP_VSA10','SlogP_VSA11','EState_VSA1','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9','EState_VSA10','VSA_EState1','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9','MID','AMID','MID_h','AMID_h','MID_C','AMID_C','MID_N','AMID_N','MID_O','AMID_O','MID_X','AMID_X','MPC2','MPC3','MPC4','MPC5','MPC6','MPC7','MPC8','MPC9','MPC10','TMPC10','piPC1','piPC2','piPC3','piPC4','piPC5','piPC6','piPC7','piPC8','piPC9','piPC10','TpiPC10','apol','bpol','nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','n9Ring','n10Ring','n11Ring','n12Ring','nG12Ring','nHRing','n3HRing','n4HRing','n5HRing','n6HRing','n7HRing','n8HRing','n9HRing','n10HRing','n11HRing','n12HRing','nG12HRing','naRing','n3aRing','n4aRing','n5aRing','n6aRing','n7aRing','n8aRing','n9aRing','n10aRing','n11aRing','n12aRing','nG12aRing','naHRing','n3aHRing','n4aHRing','n5aHRing','n6aHRing','n7aHRing','n8aHRing','n9aHRing','n10aHRing','n11aHRing','n12aHRing','nG12aHRing','nARing','n3ARing','n4ARing','n5ARing','n6ARing','n7ARing','n8ARing','n9ARing','n10ARing','n11ARing','n12ARing','nG12ARing','nAHRing','n3AHRing','n4AHRing','n5AHRing','n6AHRing','n7AHRing','n8AHRing','n9AHRing','n10AHRing','n11AHRing','n12AHRing','nG12AHRing','nFRing','n4FRing','n5FRing','n6FRing','n7FRing','n8FRing','n9FRing','n10FRing','n11FRing','n12FRing','nG12FRing','nFHRing','n4FHRing','n5FHRing','n6FHRing','n7FHRing','n8FHRing','n9FHRing','n10FHRing','n11FHRing','n12FHRing','nG12FHRing','nFaRing','n4FaRing','n5FaRing','n6FaRing','n7FaRing','n8FaRing','n9FaRing','n10FaRing','n11FaRing','n12FaRing','nG12FaRing','nFaHRing','n4FaHRing','n5FaHRing','n6FaHRing','n7FaHRing','n8FaHRing','n9FaHRing','n10FaHRing','n11FaHRing','n12FaHRing','nG12FaHRing','nFARing','n4FARing','n5FARing','n6FARing','n7FARing','n8FARing','n9FARing','n10FARing','n11FARing','n12FARing','nG12FARing','nFAHRing','n4FAHRing','n5FAHRing','n6FAHRing','n7FAHRing','n8FAHRing','n9FAHRing','n10FAHRing','n11FAHRing','n12FAHRing','nG12FAHRing','nRot','RotRatio','SLogP','SMR','TopoPSA(NO)','TopoPSA','GGI1','GGI2','GGI3','GGI4','GGI5','GGI6','GGI7','GGI8','GGI9','GGI10','JGI1','JGI2','JGI3','JGI4','JGI5','JGI6','JGI7','JGI8','JGI9','JGI10','JGT10','Diameter','Radius','TopoShapeIndex','PetitjeanIndex','Vabc','VAdjMat','MWC01','MWC02','MWC03','MWC04','MWC05','MWC06','MWC07','MWC08','MWC09','MWC10','TMWC10','SRW02','SRW03','SRW04','SRW05','SRW06','SRW07','SRW08','SRW09','SRW10','TSRW10','MW','AMW','WPath','WPol','Zagreb1','Zagreb2','mZagreb1','mZagreb2']
    my_descs = []
    # NOTE: ignore_3D=True
    calc_dummy = Calculator(descriptors, ignore_3D=True)
    for i, desc in enumerate(calc_dummy.descriptors):
        if desc.__str__()  in my_desc_names:
            my_descs.append(desc)
    calc = Calculator(my_descs, ignore_3D=True)
    df_mordred = calc.pandas(mols)
    df_mordred.to_csv(f'desc/{mol_type}_mordred_2D.csv', index=False)



def calc_mordred_3Ddescriptors(mol_type, sub=None):
    """
    Calculate 51 types Mordred 3D descriptors.
    """
    my_desc_names = ['PNSA1', 'PNSA2', 'PNSA3', 'PNSA4', 'PNSA5', 'PPSA1', 'PPSA2','PPSA3', 'PPSA4', 'PPSA5', 'DPSA1', 'DPSA2', 'DPSA3', 'DPSA4','DPSA5', 'FNSA1', 'FNSA2', 'FNSA3', 'FNSA4', 'FNSA5', 'FPSA1','FPSA2', 'FPSA3', 'FPSA4', 'FPSA5', 'WNSA1', 'WNSA2', 'WNSA3','WNSA4', 'WNSA5', 'WPSA1', 'WPSA2', 'WPSA3', 'WPSA4', 'WPSA5','RNCS', 'RPCS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'GeomDiameter','GeomRadius', 'GeomShapeIndex', 'GeomPetitjeanIndex', 'GRAV','GRAVp', 'Mor01', 'Mor01m', 'Mor01v', 'Mor01se', 'Mor01p','MOMI-X', 'MOMI-Y', 'MOMI-Z', 'PBF']
    my_descs = []
    # NOTE: ignore_3D=False
    calc_dummy = Calculator(descriptors, ignore_3D=False)
    for i, desc in enumerate(calc_dummy.descriptors):
        if desc.__str__()  in my_desc_names:
            my_descs.append(desc)
    calc = Calculator(my_descs, ignore_3D=False)

    if mol_type == 'peptide':
        mols = Chem.SDMolSupplier(f'sdf/{mol_type}_{sub}.sdf')
    else:
        mols = Chem.SDMolSupplier(f'sdf/{mol_type}.sdf')

    df_mordred = calc.pandas(mols)

    if mol_type == 'peptide':
        df_mordred.to_csv(f'desc/{mol_type}_mordred_3D_{sub}.csv', index=False)
    else:
        df_mordred.to_csv(f'desc/{mol_type}_mordred_3D.csv', index=False)



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



# def delete_similarity(x, y, threshold=0.9):
#     features_delete_R = []
#     for i in range(x.shape[1]):
#         if x.columns[i] not in features_delete_R:
#             a = x.iloc[:, i].values
#             for j in range(i):
#                 if x.columns[j] not in features_delete_R:
#                     b = x.iloc[:, j].values
#                     R = abs(pearsonr(a, b)[0])
#                     # If |R|>threshold, remove the one with lower correlation to y
#                     if ((R > threshold) and (i != j)):
#                         cor_a = abs(pearsonr(a, y)[0])
#                         cor_b = abs(pearsonr(b, y)[0])
#                         if cor_a <= cor_b:
#                             features_delete_R.append(x.columns[i])
#                         else:
#                             features_delete_R.append(x.columns[j])
#     features_delete_R = list(set(features_delete_R))
#     return features_delete_R



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


