import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


MAX_ATOMNUM = 128
# WARNING -1 can not work well
ATOM_PAD_VAL= 0


def get_atoms_mask(mols):
    atoms_mask = []
    for mol in mols:
        num_now = mol.GetNumAtoms()
        atoms_mask.append([0]*num_now + [1]*(MAX_ATOMNUM-num_now))
    return np.array(atoms_mask)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: int(x == s), allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))



def get_atoms_features(mols):
    """""
    2023/10/30 30dims
    """""
    atoms_features = []

    for mol in mols:
        feature_now = []
        for atom in mol.GetAtoms():
            attributes = []
            # Symbol: ['C', 'N', 'O', 'F', 'Br', 'S', 'Cl']
            # 8 dims
            attributes += one_of_k_encoding_unk(
                atom.GetSymbol(),
                ['C', 'N', 'O', 'F', 'Br', 'S', 'Cl', 'Other']
            )
            # CHANGED len(GetNeighbors) -> GetDegree
            # Degree: [1, 2, 3, 4]
            # 5 dims
            attributes += one_of_k_encoding_unk(
                atom.GetDegree,
                [1, 2, 3, 4, 'Other']
            )
            # TotalNumHs: [0, 1, 2, 3]
            # 5 dims
            attributes += one_of_k_encoding_unk(
                atom.GetTotalNumHs(),
                [0, 1, 2, 3, 'Other']
            )
            # FormalCharge: [0, 1, -1]
            # 4 dims
            attributes += one_of_k_encoding_unk(
                atom.GetFormalCharge(),
                [-1, 0, 1, 'Other']
            )
            # Hybridization: [rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]
            # 3 dims
            attributes += one_of_k_encoding_unk(
                atom.GetHybridization(),
                [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, 'Other']
            )
            # ChiralTag: [rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
            # 3 dims
            attributes += one_of_k_encoding_unk(
                atom.GetChiralTag(),
                [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
            )
            # 2 dims
            attributes.append(int(atom.IsInRing()))
            attributes.append(int(atom.GetIsAromatic()))

            feature_now.append(attributes)

        num_now = mol.GetNumAtoms()
        if num_now < MAX_ATOMNUM:
            # pad down
            for _ in range(MAX_ATOMNUM-num_now):
                feature_now.append([ATOM_PAD_VAL]*len(attributes))

        atoms_features.append(feature_now)

    return np.array(atoms_features)




def get_graph_distance_matrix(mols):
    distance_graph = []
    for mol in mols:
        num_now = mol.GetNumAtoms()
        matrix = Chem.rdmolops.GetDistanceMatrix(mol).tolist()

        if num_now < MAX_ATOMNUM:
            # pad right
            for i in range(len(matrix)):
                for _ in range(MAX_ATOMNUM-num_now):
                    matrix[i].append(ATOM_PAD_VAL)
            # pad down
            for _ in range(MAX_ATOMNUM-num_now):
                matrix.append([ATOM_PAD_VAL]*MAX_ATOMNUM)

        distance_graph.append(matrix)

    return np.array(distance_graph)


def get_conf_distance_matrix(mols):
    distance_conf = []
    for mol in mols:
        num_now = mol.GetNumAtoms()
        matrix = AllChem.Get3DDistanceMatrix(mol).tolist()

        if num_now < MAX_ATOMNUM:
            # pad right
            for i in range(len(matrix)):
                for _ in range(MAX_ATOMNUM-num_now):
                    matrix[i].append(ATOM_PAD_VAL)
            # pad down
            for _ in range(MAX_ATOMNUM-num_now):
                matrix.append([ATOM_PAD_VAL]*MAX_ATOMNUM)

        distance_conf.append(matrix)
    return np.round(np.array(distance_conf), 4)



def get_bond_type_matrix(mols):
    bond_types = []
    weight = [1, 2, 3, 1.5]
    for mol in mols:
        matrix = np.zeros([MAX_ATOMNUM, MAX_ATOMNUM])
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            bt = bond.GetBondType()

            bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
            # for i, m in enumerate(bond_feats):
            #     if m == True:
            #         b = weight[i]
            # NOTE: 1.4 is conjugated bond
            for i, m in enumerate(bond_feats):
                if m == True and i != 0:
                    b = weight[i]
                elif m == True and i == 0:
                    if bond.GetIsConjugated() == True:
                        b = 1.4
                    else:
                        b = 1
                else:pass

            matrix[a1, a2] = b
            matrix[a2, a1] = b
        bond_types.append(matrix)

    return np.array(bond_types)