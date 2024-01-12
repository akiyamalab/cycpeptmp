import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_ATOMNUM = 128
ATOM_PAD_VAL= 0
ATOMS_FEATURES_NUM = 30


class PositionalEncoding(nn.Module):
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
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    # self-attention block
    def _sa_block(self, x, distance, attn_mask, key_padding_mask):
        # NOTE attn_weight is average of all heads
        # x (attn_output): QKV, attn_weights: QK
        x, attn_weights = self.self_attn(x, x, x,
                                        attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask)
        # IMPORTANT
        x = x * distance
        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, distance, src_mask=None, src_key_padding_mask=None):
        x = src
        x_attn, attn_weights = self._sa_block(x, distance, src_mask, src_key_padding_mask)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self._ff_block(x))
        return x, x_attn, attn_weights



class TransformerModel(nn.Module):
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

        if activation_name == 'GELU':
            self.activation = nn.GELU()
        elif activation_name == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_name == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation_name == 'SiLU':
            self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout_rate)

        # Embedding
        self.embedding_atoms = nn.Linear(ATOMS_FEATURES_NUM, model_dim)
        if use_2D:
            self.embedding_graph = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_3D:
            self.embedding_conf  = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_bond:
            self.embedding_bond  = nn.Linear(MAX_ATOMNUM, model_dim)
        if use_abpe:
            self.add_pos = PositionalEncoding(model_dim, dropout_rate)

        # Encoder
        self.gamma_g = gamma_g
        self.gamma_c = gamma_c

        self.n_encoders = n_encoders
        if use_2D:
            self.transformer_encoder_graph = nn.Sequential()
        if use_3D:
            self.transformer_encoder_conf = nn.Sequential()

        # NOTE distanceは固定
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

            # WARNING 偷懒了，use_auxiliary的地方只有use_3D的判定
            if use_auxiliary:
                # NOTE 次元数多いため16に圧縮
                if use_3D:
                    self.auxiliary_concat_layers.add_module(f'auxiliary_concat_{i}', nn.Linear(model_dim*2, 16))
                else:
                    self.auxiliary_concat_layers.add_module(f'auxiliary_concat_{i}', nn.Linear(model_dim, 16))
                # NOTE ouput 1dim
                self.auxiliary_out_layers.add_module(f'auxiliary_out_{i}', nn.Linear(MAX_ATOMNUM*16, 1))


        # FCs
        self.linear_layers = nn.Sequential()

        # NOTE　こっちも圧縮だが32
        if use_3D and use_2D:
            self.linear_layers.add_module('linear_concat', nn.Linear(model_dim*2, 32))
        else:
            self.linear_layers.add_module('linear_concat', nn.Linear(model_dim, 32))
        self.linear_layers.add_module('ac_concat', self.activation)
        self.linear_layers.add_module('dropout_concat', self.dropout)
        self.linear_layers.add_module('flatten', nn.Flatten())

        for i in range(n_linears):
            if i == 0:
                # WARNING 重い
                # x = torch.mean(x, dim=-1) だと50s
                # いやでもこっちも57s ??
                self.linear_layers.add_module(f'linear_{i}', nn.Linear(MAX_ATOMNUM*32, dim_linear))
            else:
                self.linear_layers.add_module(f'linear_{i}', nn.Linear(dim_linear, dim_linear))
            self.linear_layers.add_module(f'ac_{i}', self.activation)
            self.linear_layers.add_module(f'dropout_{i}', self.dropout)

        self.linear_layers.add_module('linear_out', nn.Linear(dim_linear, dim_out))



    def forward(self, x_batch):
        atoms_mask__, atoms_features__, graph__, conf__, bond__ = \
            x_batch[0].to(self.device), x_batch[1].to(self.device), x_batch[2].to(self.device), x_batch[3].to(self.device), x_batch[4].to(self.device)

        # [batch_size, MAX_ATOMNUM, dims] -> [MAX_ATOMNUM, batch_size, model_dim]
        atoms_features__ = self.embedding_atoms(atoms_features__.transpose(0, 1))
        # OPTIMIZE
        if self.use_bond:
            bond__ = self.embedding_bond(bond__.transpose(0, 1))
            atoms_features__ = atoms_features__ + bond__

        # [batch_size, MAX_ATOMNUM, MAX_ATOMNUM] -> [MAX_ATOMNUM, batch_size, model_dim]
        # OPTIMIZE
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
                # NOTE do not update distance matrix
                # graph__ = self.forward_graph[i](graph__)
                # conf__  = self.forward_conf[i](conf__)

                if self.use_2D:
                    atoms_features_graph = self.transformer_encoder_graph[i](src=atoms_features_graph, distance=graph__, src_key_padding_mask=atoms_mask__)[0]
                if self.use_3D:
                    atoms_features_conf  = self.transformer_encoder_conf[i](src=atoms_features_conf, distance=conf__, src_key_padding_mask=atoms_mask__)[0]

            # WARNING 偷懒了，use_auxiliary的地方只有use_3D的判定
            if self.use_auxiliary:
                if self.use_3D:
                    auxiliary__ = torch.cat([self.gamma_g * atoms_features_graph, self.gamma_c * atoms_features_conf], dim=2).transpose(0, 1)
                else:
                    auxiliary__ = atoms_features_graph.transpose(0, 1)
                auxiliary__ = self.auxiliary_concat_layers[i](auxiliary__)
                # NOTE 如果要加ac的话不能在同一个auxiliary_concat_layers里，要init单独宣言，所以在forward里面写吧
                auxiliary__ = self.activation(auxiliary__)
                # no dropout
                # auxiliary__ = self.dropout(auxiliary__)
                auxiliary__ = nn.Flatten()(auxiliary__)
                # [batch_size, 1]
                auxiliary__ = self.auxiliary_out_layers[i](auxiliary__)
                if i == 0:
                    auxiliary_list = auxiliary__
                else:
                    auxiliary_list = torch.cat([auxiliary_list, auxiliary__], dim=1)


        # [batch_size, MAX_ATOMNUM, model_dim*2]
        if self.use_2D and self.use_3D:
            x = torch.cat([self.gamma_g * atoms_features_graph, self.gamma_c * atoms_features_conf], dim=2).transpose(0, 1)
        elif self.use_2D:
            x = atoms_features_graph.transpose(0, 1)
        elif self.use_3D:
            x = atoms_features_conf.transpose(0, 1)

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
