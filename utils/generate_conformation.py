import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem



def generate_peptide_conformation(config, df, file_path):
    """
    Only just one 3D conformation will be generated per SMILES representation, and duplicate structures are not removed by rmsd for computational cost.
    Prediction accuracy could be improved by using conformations generated in a more rigorous manner.
    """
    if not os.path.exists(file_path):

        writer = Chem.SDWriter(file_path)

        smiles = df['SMILES'].to_list()
        id = df['ID'].to_list()

        for i in tqdm(range(len(df))):
            smi = smiles[i]
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            # NOTE: useMacrocycleTorsions
            AllChem.EmbedMolecule(mol, \
                                  maxAttempts=config['conformation']['maxAttempts'], \
                                  useMacrocycleTorsions=True)
            if config['conformation']['ff'] == 'UFF':
                AllChem.UFFOptimizeMolecule(mol)
            # TODO: Add more force fields

            mol.SetProp("_Name", str(id[i]))
            writer.write(mol)

        writer.close()

    else:
        print('Peptide conformations already exists.')




def generate_monomer_conformation(config, df, file_path):
    """
    Unlike peptides, generate mono_conf_num (200) conformations per monomer and select the top replica_num (60) conformations.
    """
    if not os.path.exists(file_path):

        writer = Chem.SDWriter(file_path)
        smiles = df['SMILES'].to_list()
        id = df['ID'].to_list()

        for i in tqdm(range(len(df))):
            smi = smiles[i]
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))

            # EmbedMultipleConfs
            cids = AllChem.EmbedMultipleConfs(mol, \
                                            numConfs=config['conformation']['mono_conf_num'], \
                                            maxAttempts=config['conformation']['maxAttempts'], \
                                            randomSeed=config['conformation']['mono_seed'], \
                                            pruneRmsThresh=config['conformation']['mono_rms_thresh'], \
                                            numThreads=10, \
                                            useMacrocycleTorsions=False)

            # kcal/mol
            energy = []
            for cid in cids:
                uff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                uff.Minimize()
                energy.append((uff.CalcEnergy(), cid))
            # Check the number of conformers
            if len(energy) < config['augmentation']['replica_num']:
                print(f'Number of conformations of monomer: {id[i]} is not enough: {len(energy)}.')

            # Sort by energy
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

    else:
        print('Monomer conformations already exists.')