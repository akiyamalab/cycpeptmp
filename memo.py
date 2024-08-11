df_peptide = pd.read_csv(data_args['org_peptide_path'], low_memory=False)
df_monomer = pd.read_csv(data_args['org_monomer_path'], low_memory=False)

smiles = df_peptide['SMILES'].tolist()
shape = df_peptide['Molecule_Shape'].to_list()
helm = df_peptide['HELM'].to_list()
symbol_to_smiles = dict(zip(df_monomer['Symbol'], df_monomer['capped_SMILES']))
symbol_to_cxsmiles = dict(zip(df_monomer['Symbol'], df_monomer['CXSMILES']))
R3_dict = dict(zip(df_monomer['Symbol'], df_monomer['R3']))
smiles_to_symbol = dict(zip(df_monomer['capped_SMILES'], df_monomer['Symbol']))


substructure_list, substructure_num = [], []

for i in range(len(df_peptide)):

    now_substructure = []
    now_seq = helm[i].split('$')[0].split('{')[1].replace('}', '').replace('[', '').replace(']', '').split('.')

    if shape[i] == 'Circle':
        now_substructure = [symbol_to_smiles[_] for _ in now_seq]
    elif shape[i] == 'Lariat':
        # Lariat peptides, do not divide bonds of side chain
        atts = helm[i].split('$')[1].split(',')[2].split('-')
        atts_num = [int(_.split(':')[0]) for _ in atts]
        atts_R = [_.split(':')[1] for _ in atts]

        # HELM example of this case: PEPTIDE48{A.A.L.[meV].L.F.F.P.I.T.G.D.[-pip]}$PEPTIDE48,PEPTIDE48,1:R1-12:R3$$$
        if atts_num[0] == 1:
            # NOTE: This case were all R1-R3
            # if atts_R[0] != 'R1':
            #     print(f'{i}, 0, {atts_R[0]}')
            # elif atts_R[1] != 'R3':
            #     print(f'{i}, 1, {atts_R[1]}')

            now_substructure = [symbol_to_smiles[_] for _ in now_seq[:atts_num[1]-1]]
            # monomers to combine
            cxsmiles = [symbol_to_cxsmiles[_] for _ in now_seq[atts_num[1]-1:]]
            # NOTE: 第一个cap两处(R1, R3), side chain不cap
            tmp = cxsmiles[0].split(' |')[0]
            for _ in re.findall('_R\d', cxsmiles[0]):
                if _ == '_R1':
                    tmp = tmp.replace('[*]', '[CH3]', 1)
                elif _ == '_R2':
                    tmp = tmp.replace('[*]', '[2C]', 1)
                elif _ == '_R3':
                    if R3_dict[now_seq[atts_num[1]-1]] == 'H':
                        tmp = tmp.replace('[*]', '[CH3]', 1)
                    elif R3_dict[now_seq[atts_num[1]-1]] == 'OH':
                        tmp = tmp.replace('[*]', '[H]', 1)
            cxsmiles[0] = tmp

            combined = utils_function.combine_cxsmiles(cxsmiles, now_seq[atts_num[1]-1:], R3_dict)
            now_substructure.append(combined)

        # HELM example of this case: PEPTIDE959{[Mono22-].G.T.[Mono23].[Mono24].[dLeu(3R-OH)].[dSer(Me)].G.A.[meT].[dTyr(bR-OMe)].[Mono25]}$PEPTIDE959,PEPTIDE959,6:R3-12:R2$$$
        else:
            # NOTE: This case were all R3-R2
            # if atts_R[0] != 'R3':
            #     print(f'{i}, 0, {atts_R[0]}')
            # elif atts_R[1] != 'R2':
            #     print(f'{i}, 1, {atts_R[1]}')
            cxsmiles = [symbol_to_cxsmiles[_] for _ in now_seq[:atts_num[0]]]
            # NOTE: 最后一个cap两处(R2, R3), side chain不cap
            tmp = cxsmiles[-1].split(' |')[0]
            for _ in re.findall('_R\d', cxsmiles[-1]):
                if _ == '_R1':
                    tmp = tmp.replace('[*]', '[1C]', 1)
                elif _ == '_R2':
                    tmp = tmp.replace('[*]', '[H]', 1)
                elif _ == '_R3':
                    if R3_dict[now_seq[atts_num[0]-1]] == 'H':
                        tmp = tmp.replace('[*]', '[CH3]', 1)
                    elif R3_dict[now_seq[atts_num[0]-1]] == 'OH':
                        tmp = tmp.replace('[*]', '[H]', 1)
            cxsmiles[-1] = tmp

            combined = utils_function.combine_cxsmiles(cxsmiles, now_seq[:atts_num[0]], R3_dict)
            now_substructure.append(combined)
            now_substructure += [symbol_to_smiles[_] for _ in now_seq[atts_num[0]:]]

    substructure_num.append(len(now_substructure))
    if len(now_substructure) < data_args['monomer_max_len']:
        now_substructure += [''] * (data_args['monomer_max_len'] - len(now_substructure))
    substructure_list.append(now_substructure)

# check
df_peptide['Monomer_Length_in_Main_Chain'].to_list() == substructure_num




# Save substructure table
if not os.path.exists(data_args['substructures_table_path']):
    pd.concat([df_peptide[['CycPeptMPDB_ID', 'Source', 'Year', 'Original_Name_in_Source_Literature', \
                           'Structurally_Unique_ID', 'Same_Peptides_ID', 'SMILES', 'HELM', \
                           'Monomer_Length', 'Monomer_Length_in_Main_Chain', 'Molecule_Shape', 'Permeability', \
                           'PAMPA', 'Caco2', 'MDCK', 'RRCK']],
               pd.DataFrame(substructure_list, columns=[f'Substructure-{i}' for i in range(1, data_args['monomer_max_len']+1)])], axis=1).to_csv(data_args['substructures_table_path'], index=False)