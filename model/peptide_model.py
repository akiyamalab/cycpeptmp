import json
import torch
import torch.nn as nn


config_path = 'config/CycPeptMP.json'
config = json.load(open(config_path,'r'))
NUM_FP = config['fingerprint']['bit_num'] * 2
NUM_DESC = len(config['descriptor']['desc_2D'])+len(config['descriptor']['desc_3D'])
NUM_DESC_2D = len(config['descriptor']['desc_2D'])



class MultiLayerPerceptron(nn.Module):
    """
    MLP-based peptide model.
    """
    def __init__(self, device,
                num_mlp, dim_mlp,
                activation_name, dropout_rate,
                dim_linear,
                dim_out=1,
                use_auxiliary=False,
                use_3D=True,
                classification=False,
                ):
        super(MultiLayerPerceptron, self).__init__()

        self.use_auxiliary = use_auxiliary
        self.use_3D = use_3D
        self.device = device
        # NOTE: Use dropout, not batchnorm
        self.dropout = nn.Dropout(dropout_rate)
        self.classification = classification

        if use_3D:
            self.NUM_DESC = NUM_DESC
        else:
            self.NUM_DESC = NUM_DESC_2D

        if activation_name == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_name == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation_name == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_name == 'GELU':
            self.activation = nn.GELU()

        self.desc_layers = nn.Sequential()
        self.fp_layers = nn.Sequential()
        self.num_mlp = num_mlp
        self.dim_mlp = dim_mlp

        if use_auxiliary:
            self.auxiliary_concat_layers, self.auxiliary_out_layers = nn.Sequential(), nn.Sequential()


        for i in range(num_mlp):
            if i == 0:
                linear_desc = nn.Linear(in_features=self.NUM_DESC, out_features=dim_mlp)
                linear_fp = nn.Linear(in_features=NUM_FP, out_features=dim_mlp)
            else:
                linear_desc = nn.Linear(in_features=dim_mlp, out_features=dim_mlp)
                linear_fp = nn.Linear(in_features=dim_mlp, out_features=dim_mlp)

            self.desc_layers.add_module(f'mlp_desc_{i}', linear_desc)
            self.fp_layers.add_module(f'mlp_fp_{i}', linear_fp)

            # self.desc_layers.add_module(f'bn_mlp_desc_{i}', nn.BatchNorm1d(num_features=dim_mlp))
            # self.fp_layers.add_module(f'bn_mlp_fp_{i}', nn.BatchNorm1d(num_features=dim_mlp))

            # self.desc_layers.add_module('ac_mlp_desc_{}'.format(i), self.activation)
            # self.fp_layers.add_module('ac_mlp_fp_{}'.format(i), self.activation)

            if use_auxiliary:
                # Concat the output of the two MLPs
                self.auxiliary_concat_layers.add_module(f'auxiliary_concat_{i}', nn.Linear(dim_mlp*2, 64))
                self.auxiliary_out_layers.add_module(f'auxiliary_out_{i}', nn.Linear(64, 1))


        self.linear_layers = nn.Sequential()
        self.linear_layers.add_module('linear_concat', nn.Linear(dim_mlp*2, dim_linear))
        # self.linear_layers.add_module('bn_concat', nn.BatchNorm1d(num_features=dim_linear))
        self.linear_layers.add_module('dropout_concat', self.dropout)
        self.linear_layers.add_module('ac_concat', self.activation)

        self.linear_layers.add_module('out_mlp', nn.Linear(in_features=dim_linear, out_features=dim_out))



    def forward(self, x_batch):
        """
        Args:
            x_batch:
                desc__: peptide descriptors, shape [batch_size, NUM_DESC]
                fp__: morgan fingerprints, shape [batch_size, NUM_FP]
        Outputs:
            auxiliary_list: auxiliary loss of MLP layers
            output: peptide-level latent features, shape [batch_size, dim_out]
        """
        desc__, fp__ = x_batch[0].to(self.device), x_batch[1].to(self.device)

        if not self.use_3D:
            # Only 2D descriptors
            desc__ = desc__[:, :self.NUM_DESC]

        for i in range(self.num_mlp):
            desc__ = self.desc_layers[i](desc__)
            # desc__ = nn.BatchNorm1d(num_features=self.dim_mlp)(desc__)
            desc__ = self.dropout(desc__)
            desc__ = self.activation(desc__)

            fp__ = self.fp_layers[i](fp__)
            # fp__ = nn.BatchNorm1d(num_features=self.dim_mlp)(fp__)
            fp__ = self.dropout(fp__)
            fp__ = self.activation(fp__)

            if self.use_auxiliary:
                auxiliary__ = torch.cat([desc__, fp__], dim=1)
                auxiliary__ = self.auxiliary_concat_layers[i](auxiliary__)
                auxiliary__ = self.auxiliary_out_layers[i](auxiliary__)
                if self.classification:
                    auxiliary__ = torch.sigmoid(auxiliary__)
                if i == 0:
                    auxiliary_list = auxiliary__
                else:
                    auxiliary_list = torch.cat([auxiliary_list, auxiliary__], dim=1)

        # Equation (5) in the paper
        out = torch.cat([desc__, fp__], dim=1)
        out = self.linear_layers(out)

        if self.use_auxiliary:
            return auxiliary_list, out
        else:
            return out