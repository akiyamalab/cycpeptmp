import torch
import torch.nn as nn
import torch.nn.functional as F

import atoms_model
import monomers_model
import peptides_model


class FusionModel(nn.Module):
    def __init__(self,
            device, use_auxiliary,
            # Transformer
            Trans_activation, Trans_dropout_rate,
            Trans_n_encoders, Trans_head_num, Trans_model_dim, Trans_dim_feedforward,
            Trans_gamma_g, Trans_gamma_c,
            Trans_n_linears, Trans_dim_linear, Trans_dim_out,
            # CNN
            CNN_type, CNN_num_conv, CNN_conv_units, CNN_padding,
            CNN_activation_name, CNN_pooling_name,
            CNN_num_linear, CNN_linear_units, CNN_dim_out,
            # MLP
            MLP_num_mlp, MLP_dim_mlp,
            MLP_activation_name, MLP_dropout_rate,
            MLP_dim_linear, MLP_dim_out,
            # Fusion
            Fusion_num_concat, Fusion_concat_units,
            # ablation
            use_3D=True,
            use_atom_model=True, use_monomer_model=True, use_peptide_model=True,
            ensemble=False,
            # visualization
            visualization=False,
            ):
        super(FusionModel, self).__init__()

        self.use_atom_model = use_atom_model
        self.use_monomer_model = use_monomer_model
        self.use_peptide_model = use_peptide_model
        self.ensemble = ensemble
        self.visualization = visualization

        if ensemble:
            Trans_dim_out = 1
            CNN_dim_out   = 1
            MLP_dim_out   = 1

        if use_atom_model:
            self.atoms_model = atoms_model.TransformerModel(device,
                activation_name=Trans_activation, dropout_rate=Trans_dropout_rate,
                n_encoders=Trans_n_encoders, head_num=Trans_head_num, model_dim=Trans_model_dim, dim_feedforward=Trans_dim_feedforward,
                gamma_g=Trans_gamma_g, gamma_c=Trans_gamma_c,
                n_linears=Trans_n_linears, dim_linear=Trans_dim_linear, dim_out=Trans_dim_out,
                use_auxiliary=use_auxiliary, use_3D=use_3D, visualization=visualization)

        if use_monomer_model:
            self.monomers_model = monomers_model.ConvolutionalNeuralNetwork(device,
                cnn_type=CNN_type, num_conv=CNN_num_conv, conv_units=CNN_conv_units, padding=CNN_padding,
                activation_name=CNN_activation_name, pooling_name=CNN_pooling_name,
                num_linear=CNN_num_linear, linear_units=CNN_linear_units, dim_out=CNN_dim_out,
                use_auxiliary=use_auxiliary, use_3D=use_3D)

        if use_peptide_model:
            self.peptides_model = peptides_model.MultiLayerPerceptron(device,
                num_mlp=MLP_num_mlp, dim_mlp=MLP_dim_mlp,
                activation_name=MLP_activation_name, dropout_rate=MLP_dropout_rate,
                dim_linear=MLP_dim_linear, dim_out=MLP_dim_out, use_auxiliary=use_auxiliary, use_3D=use_3D)


        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            if use_atom_model:
                self.linear_subout_atoms = nn.Linear(in_features=Trans_dim_out, out_features=1)
            if use_monomer_model:
                self.linear_subout_monomers = nn.Linear(in_features=CNN_dim_out, out_features=1)
            if use_peptide_model:
                self.linear_subout_peptides = nn.Linear(in_features=MLP_dim_out, out_features=1)


        self.concat_layers = nn.Sequential()

        if MLP_activation_name == 'ReLU':
            activation_concat = nn.ReLU()
        elif MLP_activation_name == 'LeakyReLU':
            activation_concat = nn.LeakyReLU()
        elif MLP_activation_name == 'SiLU':
            activation_concat = nn.SiLU()
        elif MLP_activation_name == 'GELU':
            activation_concat = nn.GELU()


        for i in range(Fusion_num_concat):
            if i == 0:
                if use_atom_model and use_monomer_model and use_peptide_model:
                    linear = nn.Linear(in_features=Trans_dim_out+CNN_dim_out+MLP_dim_out, out_features=Fusion_concat_units[i])
                elif use_atom_model and use_monomer_model:
                    linear = nn.Linear(in_features=Trans_dim_out+CNN_dim_out, out_features=Fusion_concat_units[i])
                elif use_atom_model and use_peptide_model:
                    linear = nn.Linear(in_features=Trans_dim_out+MLP_dim_out, out_features=Fusion_concat_units[i])
                elif use_monomer_model and use_peptide_model:
                    linear = nn.Linear(in_features=CNN_dim_out+MLP_dim_out, out_features=Fusion_concat_units[i])
            else:
                linear = nn.Linear(in_features=Fusion_concat_units[i-1], out_features=Fusion_concat_units[i])
            self.concat_layers.add_module('concat_{}'.format(i), linear)

            # NOTE non batch normalization
            # bn_concat = nn.BatchNorm1d(num_features=Fusion_concat_units[i])
            # self.concat_layers.add_module('bn_concat_{}'.format(i), bn_concat)

            self.concat_layers.add_module('ac_concat_{}'.format(i), activation_concat)

        self.concat_layers.add_module('out_concat', nn.Linear(in_features=Fusion_concat_units[-1], out_features=1))



    def forward(self, x_batch):
        # NOTE
        x_batch_atoms = x_batch[:5]
        x_batch_monomers = x_batch[5:7]
        x_batch_peptides = x_batch[7:]

        if self.use_auxiliary:
            if self.use_atom_model:
                if self.visualization:
                    auxiliary_atom, output_atom, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf = self.atoms_model(x_batch_atoms)
                else:
                    auxiliary_atom, output_atom = self.atoms_model(x_batch_atoms)
                # [batch_size, num_encoders] -> [batch_size, 1]
                auxiliary_atom = auxiliary_atom.mean(dim=-1).unsqueeze(1)
                # [batch_size, dim_out] -> [batch_size, 1]
                auxiliary_output_atom = self.linear_subout_atoms(output_atom)
            if self.use_monomer_model:
                auxiliary_monomer, output_monomer = self.monomers_model(x_batch_monomers)
                auxiliary_monomer = auxiliary_monomer.mean(dim=-1).unsqueeze(1)
                auxiliary_output_monomer = self.linear_subout_monomers(output_monomer)
            if self.use_peptide_model:
                auxiliary_peptide, output_peptide = self.peptides_model(x_batch_peptides)
                auxiliary_peptide = auxiliary_peptide.mean(dim=-1).unsqueeze(1)
                auxiliary_output_peptide = self.linear_subout_peptides(output_peptide)

        else:
            if self.use_atom_model:
                if self.visualization:
                    output_atom, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf = self.atoms_model(x_batch_atoms)
                else:
                    output_atom = self.atoms_model(x_batch_atoms)
            if self.use_monomer_model:
                output_monomer = self.monomers_model(x_batch_monomers)
            if self.use_peptide_model:
                output_peptide = self.peptides_model(x_batch_peptides)

        # NOTE non weight
        if self.ensemble:
            output_fusion = (output_atom + output_monomer + output_peptide) / 3
        else:
            if self.use_atom_model and self.use_monomer_model and self.use_peptide_model:
                output_concat = torch.cat([output_atom, output_monomer, output_peptide], dim=-1)
            elif self.use_atom_model and self.use_monomer_model:
                output_concat = torch.cat([output_atom, output_monomer], dim=-1)
            elif self.use_atom_model and self.use_peptide_model:
                output_concat = torch.cat([output_atom, output_peptide], dim=-1)
            elif self.use_monomer_model and self.use_peptide_model:
                output_concat = torch.cat([output_monomer, output_peptide], dim=-1)
            output_fusion = self.concat_layers(output_concat)



        if self.use_auxiliary:
            if self.use_atom_model and self.use_monomer_model and self.use_peptide_model:
                if self.visualization:
                    return auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf
                else:
                    return auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion
            elif self.use_atom_model and self.use_monomer_model:
                return auxiliary_atom, auxiliary_monomer, auxiliary_output_atom, auxiliary_output_monomer, output_fusion
            elif self.use_atom_model and self.use_peptide_model:
                return auxiliary_atom, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_peptide, output_fusion
            elif self.use_monomer_model and self.use_peptide_model:
                return auxiliary_monomer, auxiliary_peptide, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion
        else:
            return output_fusion