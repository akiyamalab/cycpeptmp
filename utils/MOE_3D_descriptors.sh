#!/bin/bash
#SBATCH -N 1
#SBATCH -J moe_desc
#SBATCH -o log/stdout.%J
#SBATCH -e log/stderr.%J

$moebatch -exec "db_Open ['../desc/peptide_moe_3D_$1.mdb', 'create'];" -exit
$moebatch -exec "db_ImportSD ['../desc/peptide_moe_3D_$1.mdb', '../sdf/peptide_$1.sdf', 'mol'];" -exit
$moebatch -exec "PartialChargeMDB ['../desc/peptide_moe_3D_$1.mdb', 'FF', 'mol', 'mol_charged'];" -exit
$moebatch -exec "QuaSAR_DescriptorMDB ['../desc/peptide_moe_3D_$1.mdb', 'mol_charged', ['ASA','ASA+','ASA-','ASA_H','ASA_P','CASA+','CASA-','DASA','DCASA','dens','dipole','E','E_ang','E_ele','E_nb','E_oop','E_sol','E_stb','E_str','E_strain','E_tor','E_vdw','FASA+','FASA-','FASA_H','FASA_P','FCASA+','FCASA-','glob','npr1','npr2','pmi','pmi1','pmi2','pmi3','rgyr','std_dim1','std_dim2','std_dim3','vol','VSA','vsurf_A','vsurf_CP','vsurf_CW1','vsurf_CW2','vsurf_CW3','vsurf_CW4','vsurf_CW5','vsurf_CW6','vsurf_CW7','vsurf_CW8','vsurf_D1','vsurf_D2','vsurf_D3','vsurf_D4','vsurf_D5','vsurf_D6','vsurf_D7','vsurf_D8','vsurf_DD12','vsurf_DD13','vsurf_DD23','vsurf_DW12','vsurf_DW13','vsurf_DW23','vsurf_EDmin1','vsurf_EDmin2','vsurf_EDmin3','vsurf_EWmin1','vsurf_EWmin2','vsurf_EWmin3','vsurf_G','vsurf_HB1','vsurf_HB2','vsurf_HB3','vsurf_HB4','vsurf_HB5','vsurf_HB6','vsurf_HB7','vsurf_HB8','vsurf_HL1','vsurf_HL2','vsurf_ID1','vsurf_ID2','vsurf_ID3','vsurf_ID4','vsurf_ID5','vsurf_ID6','vsurf_ID7','vsurf_ID8','vsurf_IW1','vsurf_IW2','vsurf_IW3','vsurf_IW4','vsurf_IW5','vsurf_IW6','vsurf_IW7','vsurf_IW8','vsurf_R','vsurf_S','vsurf_V','vsurf_W1','vsurf_W2','vsurf_W3','vsurf_W4','vsurf_W5','vsurf_W6','vsurf_W7','vsurf_W8','vsurf_Wp1','vsurf_Wp2','vsurf_Wp3','vsurf_Wp4','vsurf_Wp5','vsurf_Wp6','vsurf_Wp7','vsurf_Wp8']];" -exit