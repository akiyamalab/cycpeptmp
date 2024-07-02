from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem



def peptide_conformation_genetation(config, df, mol_type, sub):
    """
    Only just one 3D conformation will be generated per SMILES representation, and duplicate structures are not removed by rmsd for computational cost.
    Prediction accuracy could be improved by using conformations generated in a more rigorous manner.
    """
    writer = Chem.SDWriter(f'sdf/{mol_type}_{sub}.sdf')

    smiles = df['SMILES'].to_list()
    id = df['ID'].to_list()

    for i in tqdm(range(len(df))):
        smi = smiles[i]
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        # useMacrocycleTorsions
        AllChem.EmbedMolecule(mol, \
                              maxAttempts=config['conformation']['maxAttempts'], \
                              useMacrocycleTorsions=True)
        if config['conformation']['ff'] == 'UFF':
            AllChem.UFFOptimizeMolecule(mol)

        mol.SetProp("_Name", str(id[i]))
        writer.write(mol)

    writer.close()




def monomer_conformation_genetation(config, df, mol_type):
    """
    Unlike peptides, generate mono_conf_num (200) conformations per monomer and select the top replica_num (60) conformations.
    """
    writer = Chem.SDWriter(f'sdf/{mol_type}.sdf')
    smiles = df['SMILES'].to_list()
    id = df['ID'].to_list()

    for i in tqdm(range(len(df))):
        smi = smiles[i]
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))

        # EmbedMultipleConfs
        cids = AllChem.EmbedMultipleConfs(mol, \
                                          numConfs=config['conformation']['mono_conf_num'], \
                                          maxAttempts=config['conformation']['maxAttempts'], \
                                          randomSeed=1370, \
                                          pruneRmsThresh=config['conformation']['mono_rms_thresh'], \
                                          numThreads=10, \
                                          useMacrocycleTorsions=False)

        # kcal/mol
        energy = []
        for cid in cids:
            uff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            uff.Minimize()
            energy.append((uff.CalcEnergy(), cid))
        # check the number of conformers
        if len(energy) < config['augmentation']['replica_num']:
            print(f'{id[i]}  {len(energy)}')

        # sort by energy
        energy.sort()
        energy = energy[:config['augmentation']['replica_num']]
        relative_energy = [(i-energy[0][0],j) for i,j in energy]

        for j in range(config['augmentation']['replica_num']):
            cid = energy[j][1]
            mol.SetProp("_Name", str(id[i])+'_'+str(j))
            mol.SetProp('cid', str(cid))
            mol.SetProp('Energy', str(energy[j][0]))
            mol.SetProp('Relative energy', str(relative_energy[j][0]))
            writer.write(mol, confId=cid)

    writer.close()





# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Generate and optimize conformer using RDKit.')
#     parser.add_argument('arg1', help='index of the subset (0~', type=int)
#     parser.add_argument('--sub_len', help='length of each subset', type=int)
#     # parser.add_argument('--rmsd', help='redundant conformation threshold', default=1.0, type=float)
#     # parser.add_argument('--num_initial', help='number of conformers generated before selecting by energy', default=5000, type=int)
#     # parser.add_argument('--num_conf', help='number of conformations to generate', default=500, type=int)
#     parser.add_argument('--maxAttempts', help='maximum number of generation attempts', type=int)
#     # parser.add_argument('--ff', help='optimization method', default='uff', choices=['uff', 'mmff'])

#     args = parser.parse_args()

#     # peptide_conformation_genetation(args)


