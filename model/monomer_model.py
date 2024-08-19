import json
import torch
import torch.nn as nn
import torch.nn.functional as F


config_path = 'config/CycPeptMP.json'
config = json.load(open(config_path,'r'))
SUB_MAX_LEN = config['data']['mono_max_len']
SUB_PAD_ID = config['data']['mono_pad_id']
SUB_PAD_VAL = config['data']['mono_pad_val']
NUM_DESC = len(config['descriptor']['desc_2D'])+len(config['descriptor']['desc_3D'])
NUM_DESC_2D = len(config['descriptor']['desc_2D'])


class ConvolutionalNeuralNetwork(nn.Module):
    """
    1D CNN-based monomer model.
    """
    def __init__(self, device,
                cnn_type, num_conv, conv_units, padding,
                activation_name, pooling_name,
                num_linear, linear_units,
                k_size=3, use_mask=True,
                dim_out=1,
                use_auxiliary=False,
                use_3D=True,
                classification=False,
                ):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.device = device
        self.use_auxiliary = use_auxiliary
        self.cnn_type = cnn_type
        self.use_mask = use_mask
        self.use_3D = use_3D
        self.classification = classification

        if use_3D:
            self.NUM_DESC = NUM_DESC
        else:
            self.NUM_DESC = NUM_DESC_2D

        # Convolutional layers
        self.conv_layers = nn.Sequential()
        if use_auxiliary:
            self.auxiliary_layers, self.auxiliary_out_layers = nn.Sequential(), nn.Sequential()

        for i in range(num_conv):
            if i == 0:
                conv = nn.Conv1d(in_channels=self.NUM_DESC, out_channels=conv_units[i], kernel_size=k_size, padding=padding)
            else:
                conv = nn.Conv1d(in_channels=conv_units[i-1], out_channels=conv_units[i], kernel_size=k_size, padding=padding)
            self.conv_layers.add_module(f'conv1d_{i}', conv)

            self.conv_layers.add_module(f'bn_conv_{i}', nn.BatchNorm1d(num_features=conv_units[i]))

            # NOTE： Without a duplicate declare will get an error
            if activation_name == 'ReLU':
                self.activation = nn.ReLU()
            elif activation_name == 'LeakyReLU':
                self.activation = nn.LeakyReLU()
            elif activation_name == 'SiLU':
                self.activation = nn.SiLU()
            elif activation_name == 'GELU':
                self.activation = nn.GELU()

            self.conv_layers.add_module(f'ac_conv_{i}', self.activation)

            if use_auxiliary:
                self.auxiliary_layers.add_module(f'auxiliary_{i}', nn.Linear(conv_units[i], 16))
                self.auxiliary_out_layers.add_module(f'auxiliary_out_{i}', nn.Linear(SUB_MAX_LEN*16, 1))


        # Pooling & Linear layers
        self.linear_layers = nn.Sequential()

        if pooling_name == 'ave':
            pool = nn.AvgPool1d(kernel_size=2)
        elif pooling_name == 'max':
            pool = nn.MaxPool1d(kernel_size=2)

        self.linear_layers.add_module('pooling', pool)
        self.linear_layers.add_module('flatten', nn.Flatten())

        for i in range(num_linear):
            if i == 0:
                if padding or self.cnn_type=='AugCyclicConv':
                    linear = nn.Linear(in_features=int(int(SUB_MAX_LEN/2)*conv_units[-1]), \
                                       out_features=linear_units[i])
                else:
                    linear = nn.Linear(in_features=int(int((SUB_MAX_LEN-((k_size-1)*num_conv))/2)*conv_units[-1]), \
                                       out_features=linear_units[i])
            else:
                linear = nn.Linear(in_features=linear_units[i-1], out_features=linear_units[i])
            self.linear_layers.add_module(f'convlinear_{i}', linear)

            self.linear_layers.add_module(f'bn_convlinear_{i}', nn.BatchNorm1d(num_features=linear_units[i]))
            self.linear_layers.add_module(f'ac_convlinear_{i}', self.activation)

        self.linear_layers.add_module('out_conv', nn.Linear(in_features=linear_units[-1], out_features=dim_out))



    def compensate_terminal(self, feature_map, table):
        """
        Implementation of CyclicConv.
        """
        compensated_map = torch.zeros(feature_map.shape[0], feature_map.shape[1], feature_map.shape[2]+2).to(torch.float32).to(self.device)

        # WARNING: Calculation is extremely slow....
        for i in range(len(table)):
            non_padding_indices = torch.nonzero(table[i] != SUB_PAD_ID)
            start_idx = non_padding_indices[0].item()
            end_idx = non_padding_indices[-1].item()

            # descriptors_start = feature_map[i, :, start_idx].view(-1, 1)
            # descriptors_sequence = feature_map[i, :, start_idx:end_idx+1]
            # descriptors_end = feature_map[i, :, end_idx].view(-1, 1)
            # padding_start = feature_map[i, :, :start_idx]
            # padding_end = feature_map[i, :, end_idx+1:]
            # compensated_map[i] = torch.cat([padding_start, descriptors_end, descriptors_sequence, descriptors_start, padding_end], dim=-1)

            descriptors_sequence = feature_map[i, :, start_idx:end_idx+1]
            descriptors_sequence = F.pad(descriptors_sequence.unsqueeze(0) , (1, 1), mode='circular')
            descriptors_sequence = F.pad(descriptors_sequence, (start_idx, SUB_MAX_LEN-end_idx-1), mode='constant', value=SUB_PAD_VAL)
            compensated_map[i] = descriptors_sequence.squeeze(0)

        return compensated_map



    def mask_fill(self, x, table):
        """
        Sequence masking.
        """
        mask = table == SUB_PAD_ID
        masked_x = x.clone().masked_fill_(mask.unsqueeze(1), SUB_PAD_VAL)
        return masked_x



    def forward(self, x_batch):
        """
        Args:
            x_batch:
                table: monomer sequence, shape [batch_size, SUB_MAX_LEN]
                x: monomer feature maps, shape [batch_size, NUM_DESC, SUB_MAX_LEN]
        Outputs:
            auxiliary_list: auxiliary loss of convolutional layers
            output: monomer-level latent features, shape [batch_size, dim_out]
        """
        table, x = x_batch[0].to(self.device), x_batch[1].to(self.device)

        if not self.use_3D:
            # only 2D descriptors
            x = x[:, :self.NUM_DESC, :]


        i = 0
        for name, layer in self.conv_layers.named_children():

            # NOTE: type of convolutional layers is determined by the Optuna tuning
            # before convolution
            if 'conv1d_' in name:
                if self.cnn_type == 'AugCyclicConv':
                    x = self.compensate_terminal(x, table)
                else:
                    # mask
                    if self.use_mask:
                        x = self.mask_fill(x, table)

            # convolution
            x = layer(x)

            # after convolution
            if 'ac_conv_' in name:
                # BUG: if don't declare duplicately ac_conv_ in for i in range(num_conv): will get an error, but actually it's correct...
                if self.use_auxiliary:
                    # shape [batch_size, conv_units[i], SUB_MAX_LEN] -> [batch_size, SUB_MAX_LEN, conv_units[i]]
                    auxiliary__ = x.transpose(1, 2)
                    auxiliary__ = self.auxiliary_layers[i](auxiliary__)
                    auxiliary__ = self.activation(auxiliary__)
                    auxiliary__ = nn.Flatten()(auxiliary__)
                    auxiliary__ = self.auxiliary_out_layers[i](auxiliary__)
                    if self.classification:
                        auxiliary__ = torch.sigmoid(auxiliary__)
                    if i == 0:
                        auxiliary_list = auxiliary__
                    else:
                        auxiliary_list = torch.cat([auxiliary_list, auxiliary__], dim=1)
                    i += 1

        output = self.linear_layers(x)

        if self.use_auxiliary:
            return auxiliary_list, output
        else:
            return output