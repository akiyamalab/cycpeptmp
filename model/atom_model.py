import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_ATOMNUM = 128
ATOM_PAD_VAL= 0
ATOMS_FEATURES_NUM = 30


class PositionalEncoding(nn.Module):
    """
    Original absolute positional encoding for ablation study.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [MAX_ATOMNUM, batch_size, model_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




class TransformerEncoderLayer(nn.Module):
    """
    Encoder block of the transformer model.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation


    def _sa_block(self, x, distance, attn_mask, key_padding_mask):
        """
        # Multi-head self-attention block.
        """
        # NOTE: attn_weight is average of all heads, x (attn_output): QKV, attn_weights: QK
        x, attn_weights = self.self_attn(x, x, x,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask)
        # IMPORTANT: structural focused attention, equation (3d) in the paper
        x = x * distance
        return self.dropout1(x), attn_weights


    def _ff_block(self, x):
        """
        Feedforward block.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


    def forward(self, src, distance, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: latent features of the graph/conf block in (l-1)-th layers, shape [MAX_ATOMNUM, batch_size, model_dim]
            distance: distance maps, processed through an attenuation function to weaken distant interactions, shape [MAX_ATOMNUM, batch_size, model_dim]
        Outputs:
            x: updated latent features in l-th layers, shape [MAX_ATOMNUM, batch_size, model_dim]
            x_attn (QKV) and attn_weights (QK) can use for attention visualization.
        """
        x = src
        x_attn, attn_weights = self._sa_block(x, distance, src_mask, src_key_padding_mask)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self._ff_block(x))
        return x, x_attn, attn_weights




class TransformerModel(nn.Module):
    """
    Atom-level transformer model.
    """
    def __init__(
            self, device,
            activation_name, dropout_rate,
            n_encoders, head_num, model_dim, dim_feedforward,
            gamma_g, gamma_c,
            n_linears, dim_linear,
            dim_out=1,
            use_auxiliary=False,
            use_bond=True,
            use_abpe=False,
            use_2D=True,
            use_3D=True,
            visualization=False,
            classification=False,
            ):
        super(TransformerModel, self).__init__()

        self.trans_type = 'Transformer'
        self.device = device
        self.model_dim = model_dim
        self.use_auxiliary = use_auxiliary
        self.use_bond = use_bond
        self.use_abpe = use_abpe
        self.use_2D = use_2D
        self.use_3D = use_3D
        self.visualization = visualization
        self.classification = classification

        if activation_name == 'GELU':
            self.activation = nn.GELU()
        elif activation_name == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_name == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation_name == 'SiLU':
            self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout_rate)

        # Embedding layers
        self.embedding_atoms = nn.Linear(ATOMS_FEATURES_NUM, model_dim)
        if use_2D: # graph matrix
            self.embedding_graph = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_3D: # conf matrix
            self.embedding_conf  = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_bond: # bond matrix
            self.embedding_bond  = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_abpe: # absolute positional encoding for ablation study
            self.add_pos = PositionalEncoding(model_dim, dropout_rate)

        # Encoder blocks
        self.gamma_g = gamma_g
        self.gamma_c = gamma_c

        self.n_encoders = n_encoders
        if use_2D:
            self.transformer_encoder_graph = nn.Sequential()
        if use_3D:
            self.transformer_encoder_conf = nn.Sequential()

        # NOTE: distance matrix will not be updated
        # self.forward_graph, self.forward_conf = nn.Sequential(), nn.Sequential()

        if use_auxiliary:
            self.auxiliary_concat_layers, self.auxiliary_out_layers = nn.Sequential(), nn.Sequential()

        for i in range(n_encoders):
            if use_2D:
                self.transformer_encoder_graph.add_module(f'encoder_graph_{i}', \
                                                          TransformerEncoderLayer(d_model=model_dim, nhead=head_num, dim_feedforward=dim_feedforward, \
                                                                                  dropout=dropout_rate, activation=self.activation))
            if use_3D:
                self.transformer_encoder_conf.add_module(f'encoder_conf_{i}', \
                                                         TransformerEncoderLayer(d_model=model_dim, nhead=head_num, dim_feedforward=dim_feedforward, \
                                                                                 dropout=dropout_rate, activation=self.activation))
            # self.forward_graph.add_module(f'forward_graph_{i}', nn.Linear(model_dim, model_dim))
            # self.forward_conf.add_module(f'forward_conf_{i}', nn.Linear(model_dim, model_dim))

            if use_auxiliary:
                # NOTE: reduce the dimension
                if use_3D:
                    self.auxiliary_concat_layers.add_module(f'auxiliary_concat_{i}', nn.Linear(model_dim*2, 16))
                else:
                    self.auxiliary_concat_layers.add_module(f'auxiliary_concat_{i}', nn.Linear(model_dim, 16))

                self.auxiliary_out_layers.add_module(f'auxiliary_out_{i}', nn.Linear(MAX_ATOMNUM*16, 1))


        # FCs
        self.linear_layers = nn.Sequential()

        # NOTE: reduce the dimension
        if use_3D and use_2D:
            self.linear_layers.add_module('linear_concat', nn.Linear(model_dim*2, 32))
        else:
            self.linear_layers.add_module('linear_concat', nn.Linear(model_dim, 32))
        self.linear_layers.add_module('ac_concat', self.activation)
        self.linear_layers.add_module('dropout_concat', self.dropout)
        self.linear_layers.add_module('flatten', nn.Flatten())

        for i in range(n_linears):
            if i == 0:
                # WARNING: Calculation is heavy, 128x32 -> 32
                self.linear_layers.add_module(f'linear_{i}', nn.Linear(MAX_ATOMNUM*32, dim_linear))
            else:
                self.linear_layers.add_module(f'linear_{i}', nn.Linear(dim_linear, dim_linear))
            self.linear_layers.add_module(f'ac_{i}', self.activation)
            self.linear_layers.add_module(f'dropout_{i}', self.dropout)

        self.linear_layers.add_module('linear_out', nn.Linear(dim_linear, dim_out))



    def forward(self, x_batch):
        """
        Args:
            x_batch:
                atoms_mask: mask for atoms, shape [batch_size, MAX_ATOMNUM]
                atoms_features: node features, shape [batch_size, MAX_ATOMNUM, ATOMS_FEATURES_NUM]
                graph: distance calculated from graph representation, shape [batch_size, MAX_ATOMNUM, MAX_ATOMNUM]
                conf: euclidean distance calculated from 3D conformation, shape [batch_size, MAX_ATOMNUM, MAX_ATOMNUM]
                bond: bond type, Single(1.0), Double(2.0), Triple(3.0), Aromatic(1.5), Conjugated(1.4) and No-bond(0), shape [batch_size, MAX_ATOMNUM, MAX_ATOMNUM]
        Outputs:
            auxiliary_list: auxiliary loss of encoder blocks
            output: atom-level latent features, shape [batch_size, dim_out]
            x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf can use for attention visualization
        """
        atoms_mask__, atoms_features__, graph__, conf__, bond__ = \
            x_batch[0].to(self.device), x_batch[1].to(self.device), x_batch[2].to(self.device), x_batch[3].to(self.device), x_batch[4].to(self.device)

        # shape [batch_size, MAX_ATOMNUM, dims] -> [MAX_ATOMNUM, batch_size, model_dim]
        atoms_features__ = self.embedding_atoms(atoms_features__.transpose(0, 1))

        # NOTE: Relative positional encoding by bond matrix, equation (1) in the paper
        if self.use_bond:
            bond__ = self.embedding_bond(bond__.transpose(0, 1))
            atoms_features__ = atoms_features__ + bond__


        # NOTE: distance maps, equation (2) in the paper
        # shape [batch_size, MAX_ATOMNUM, MAX_ATOMNUM] -> [MAX_ATOMNUM, batch_size, model_dim]
        eye = torch.eye(MAX_ATOMNUM).to(self.device)

        if self.use_2D:
            graph__ = graph__ + eye
            graph__ = torch.reciprocal(graph__)
            graph__[graph__==float('inf')] = 0
            graph__ = self.embedding_graph(graph__.transpose(0, 1))

        if self.use_3D:
            conf__ = conf__ + eye
            conf__ = torch.reciprocal(conf__)
            conf__[conf__==float('inf')] = 0
            conf__ = self.embedding_conf(conf__.transpose(0, 1))

        atoms_features__ = atoms_features__ * math.sqrt(self.model_dim)

        if self.use_abpe:
            atoms_features__ = self.add_pos(atoms_features__)

        if self.use_2D:
            graph__ = graph__ * math.sqrt(self.model_dim)
        if self.use_3D:
            conf__ = conf__ * math.sqrt(self.model_dim)


        for i in range(self.n_encoders):
            if i == 0:
                if self.use_2D:
                    atoms_features_graph, x_attn_graph, attn_weights_graph = self.transformer_encoder_graph[i](src=atoms_features__, distance=graph__, src_key_padding_mask=atoms_mask__)
                if self.use_3D:
                    atoms_features_conf, x_attn_conf, attn_weights_conf  =self.transformer_encoder_conf[i](src=atoms_features__, distance=conf__, src_key_padding_mask=atoms_mask__)
            else:
                # # NOTE: distance matrix will not be updated
                # graph__ = self.forward_graph[i](graph__)
                # conf__  = self.forward_conf[i](conf__)

                if self.use_2D:
                    atoms_features_graph = self.transformer_encoder_graph[i](src=atoms_features_graph, distance=graph__, src_key_padding_mask=atoms_mask__)[0]
                if self.use_3D:
                    atoms_features_conf  = self.transformer_encoder_conf[i](src=atoms_features_conf, distance=conf__, src_key_padding_mask=atoms_mask__)[0]

            if self.use_auxiliary:
                if self.use_3D:
                    auxiliary__ = torch.cat([self.gamma_g * atoms_features_graph, self.gamma_c * atoms_features_conf], dim=2).transpose(0, 1)
                else:
                    auxiliary__ = atoms_features_graph.transpose(0, 1)
                auxiliary__ = self.auxiliary_concat_layers[i](auxiliary__)
                # NOTE: activation can't be added in the same auxiliary_concat_layers, it needs a separate declaration in init, so write it in the forward
                auxiliary__ = self.activation(auxiliary__)
                # No dropout
                # auxiliary__ = self.dropout(auxiliary__)
                auxiliary__ = nn.Flatten()(auxiliary__)
                # shape [batch_size, 1]
                auxiliary__ = self.auxiliary_out_layers[i](auxiliary__)
                if self.classification:
                    auxiliary__ = torch.sigmoid(auxiliary__)
                if i == 0:
                    auxiliary_list = auxiliary__
                else:
                    auxiliary_list = torch.cat([auxiliary_list, auxiliary__], dim=1)


        # equation (4) in the paper, shape [batch_size, MAX_ATOMNUM, model_dim*2]
        if self.use_2D and self.use_3D:
            x = torch.cat([self.gamma_g * atoms_features_graph, self.gamma_c * atoms_features_conf], dim=2).transpose(0, 1)
        elif self.use_2D:
            x = atoms_features_graph.transpose(0, 1)
        elif self.use_3D:
            x = atoms_features_conf.transpose(0, 1)

        # FCs
        output = self.linear_layers(x)

        if self.use_auxiliary:
            if self.visualization:
                return auxiliary_list, output, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf
            else:
                return auxiliary_list, output
        else:
            if self.visualization:
                return output, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf
            else:
                return output
