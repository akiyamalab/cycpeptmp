import numpy as np
import torch
import time
import random
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from scipy.stats import pearsonr

UPPER_LIMIT = -4
LOWER_LIMIT = -8



class DataSet:
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]




def load_dataset(model_type, replica_num, index_name, _10cv=False):
    if _10cv: # For 10 cv
        load_node = np.load(f'../input/Trans/{replica_num}/10cv/node_{replica_num}_{index_name}.npz')
        load_graph = np.load(f'../input/Trans/{replica_num}/10cv/graph_{replica_num}_{index_name}.npz')
        load_conf = np.load(f'../input/Trans/{replica_num}/10cv/conf_{replica_num}_{index_name}.npz')
        load_bond = np.load(f'../input/Trans/{replica_num}/10cv/bond_{replica_num}_{index_name}.npz')
        load_monomers = np.load(f'../input/CNN/{replica_num}/10cv/feature_map_{replica_num}_{index_name}.npz')
        load_peptides = np.load(f'../input/MLP/{replica_num}/10cv/peptide_{replica_num}_{index_name}.npz')
    else: # For test set
        load_node = np.load(f'../input/Trans/{replica_num}/node_{replica_num}_{index_name}.npz')
        load_graph = np.load(f'../input/Trans/{replica_num}/graph_{replica_num}_{index_name}.npz')
        load_conf = np.load(f'../input/Trans/{replica_num}/conf_{replica_num}_{index_name}.npz')
        load_bond = np.load(f'../input/Trans/{replica_num}/bond_{replica_num}_{index_name}.npz')
        load_monomers = np.load(f'../input/CNN/{replica_num}/feature_map_{replica_num}_{index_name}.npz')
        load_peptides = np.load(f'../input/MLP/{replica_num}/peptide_{replica_num}_{index_name}.npz')

    if model_type == 'Trans':
        dataset_now = DataSet(list(zip(torch.Tensor(load_peptides['id'].reshape(-1,1)).to(torch.int32),
                                       torch.Tensor(load_node['atoms_mask']).to(torch.float32),
                                       torch.Tensor(load_node['atoms_features']).to(torch.float32),
                                       torch.Tensor(load_graph['graph']).to(torch.float32),
                                       torch.Tensor(load_conf['conf']).to(torch.float32),
                                       torch.Tensor(load_bond['bond']).to(torch.float32),
                                       torch.Tensor(load_peptides['y'].reshape(-1,1)).to(torch.float32))))
    elif model_type == 'CNN':
        dataset_now = DataSet(list(zip(torch.Tensor(load_peptides['id'].reshape(-1,1)).to(torch.int32),
                                       torch.Tensor(load_monomers['table']).to(torch.int32),
                                       torch.Tensor(load_monomers['feature_map']).to(torch.float32),
                                       torch.Tensor(load_peptides['y'].reshape(-1,1)).to(torch.float32))))
    elif model_type == 'MLP':
        dataset_now = DataSet(list(zip(torch.Tensor(load_peptides['id'].reshape(-1,1)).to(torch.int32),
                                       torch.Tensor(load_peptides['peptide_descriptor']).to(torch.float32),
                                       torch.Tensor(load_peptides['fps']).to(torch.float32),
                                       torch.Tensor(load_peptides['y'].reshape(-1,1)).to(torch.float32))))
    elif model_type == 'Fusion':
        dataset_now = DataSet(list(zip(torch.Tensor(load_peptides['id'].reshape(-1,1)).to(torch.int32),
                                       torch.Tensor(load_node['atoms_mask']).to(torch.float32),
                                       torch.Tensor(load_node['atoms_features']).to(torch.float32),
                                       torch.Tensor(load_graph['graph']).to(torch.float32),
                                       torch.Tensor(load_conf['conf']).to(torch.float32),
                                       torch.Tensor(load_bond['bond']).to(torch.float32),
                                       torch.Tensor(load_monomers['table']).to(torch.int32),
                                       torch.Tensor(load_monomers['feature_map']).to(torch.float32),
                                       torch.Tensor(load_peptides['peptide_descriptor']).to(torch.float32),
                                       torch.Tensor(load_peptides['fps']).to(torch.float32),
                                       torch.Tensor(load_peptides['y'].reshape(-1,1)).to(torch.float32))))
    return dataset_now




class EarlyStopping:
    """
    Early stopping to stop the training when the validation loss does not improve after patience epochs.
    """
    def __init__(self, patience, path, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path


    def __call__(self, val_loss, epoch, model, optimizer, loss_train_list, loss_valid_list):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, epoch, model, optimizer, loss_train_list, loss_valid_list)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, epoch, model, optimizer, loss_train_list, loss_valid_list)
            self.counter = 0


    def checkpoint(self, val_loss, epoch, model, optimizer, loss_train_list, loss_valid_list):
        # Save model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_now': val_loss,
                    'loss_train_list': loss_train_list,
                    'loss_valid_list': loss_valid_list,
                    }, self.path)
        self.val_loss_min = val_loss





def train_epoch(device, epoch, model, dataloader_train, optimizer, criterion, scheduler, verbose=False,
                # For auxiliary loss
                use_auxiliary=False, gamma_layer=None, gamma_subout=None,
                # For ablation study
                use_atom_model=True, use_monomer_model=True, use_peptide_model=True, ensemble=False,
                # For attention visualization
                visualization=False,
                # For classification task
                classification=False,
                ):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for batch, x_batch in enumerate(dataloader_train):
        id__ = x_batch[0]
        logPapp__ = x_batch[-1]

        # NOTE: Generally, logPexp >= -6 is considered as high permeability
        if classification:
            logPapp__ = (logPapp__ >= -6).float()

        logPapp__ = logPapp__.to(device)
        x_batch = x_batch[1:-1]

        optimizer.zero_grad()

        if use_auxiliary:
            if use_atom_model and use_monomer_model and use_peptide_model:
                if visualization:
                    auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf = model(x_batch)
                else:
                    auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion = model(x_batch)
            elif use_atom_model and use_monomer_model:
                auxiliary_atom, auxiliary_monomer, auxiliary_output_atom, auxiliary_output_monomer, output_fusion = model(x_batch)
            elif use_atom_model and use_peptide_model:
                auxiliary_atom, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_peptide, output_fusion = model(x_batch)
            elif use_monomer_model and use_peptide_model:
                auxiliary_monomer, auxiliary_peptide, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion = model(x_batch)

            if use_atom_model:
                loss_atom = criterion(auxiliary_atom, logPapp__)
                loss_output_atom = criterion(auxiliary_output_atom, logPapp__)
            else:
                loss_atom = 0.0
                loss_output_atom = 0.0

            if use_monomer_model:
                loss_monomer = criterion(auxiliary_monomer, logPapp__)
                loss_output_monomer = criterion(auxiliary_output_monomer, logPapp__)
            else:
                loss_monomer = 0.0
                loss_output_monomer = 0.0

            if use_peptide_model:
                loss_peptide = criterion(auxiliary_peptide, logPapp__)
                loss_output_peptide = criterion(auxiliary_output_peptide, logPapp__)
            else:
                loss_peptide = 0.0
                loss_output_peptide = 0.0

            loss_output_fusion = criterion(output_fusion, logPapp__)

            if ensemble:
                loss__ = gamma_layer*loss_atom + gamma_layer*loss_monomer + gamma_layer*loss_peptide + loss_output_fusion
            # NOTE: Using auxiliary loss for training, equation (7) in the paper
            else:
                loss__ = gamma_layer*loss_atom + gamma_layer*loss_monomer + gamma_layer*loss_peptide + gamma_subout*loss_output_atom + gamma_subout*loss_output_monomer + gamma_subout*loss_output_peptide + loss_output_fusion
        else:
            pred__ = model(x_batch)
            loss__ = criterion(pred__, logPapp__)

        loss__.backward()
        optimizer.step()
        running_loss += loss__.item()

        if verbose:
            if (batch+1)%100 == 0:
                print(f'batch {batch+1:>5d}/{len(dataloader_train):>5d} of epoch {epoch:>2d} completed')

    scheduler.step()
    running_loss = running_loss / len(dataloader_train)
    return running_loss





def predict_valid(device, model, dataloader_valid, criterion,
                  # Flag for training or validation
                  istrain=True,
                  # For auxiliary loss
                  use_auxiliary=False, gamma_layer=None, gamma_subout=None,
                  # For ablation study
                  use_atom_model=True, use_monomer_model=True, use_peptide_model=True, ensemble=False,
                  # For attention visualization
                  visualization=False,
                  # For classification task
                  classification=False,
                  ):
    """
    Predict the validation/test set.
    Use istrain to distinguish between calculating loss for training and evaluation process.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        if not istrain:
            ids, exps, preds, x_attn_graphs, x_attn_confs, attn_weights_graphs, attn_weights_confs = [], [], [], [], [], [], []
        for batch, x_batch in enumerate(dataloader_valid):
            id__ = x_batch[0]
            logPapp__ = x_batch[-1]
            if classification:
                logPapp__ = (logPapp__ >= -6).float()
            logPapp__ = logPapp__.to(device)
            x_batch = x_batch[1:-1]

            if use_auxiliary:
                if use_atom_model and use_monomer_model and use_peptide_model:
                    if visualization:
                        auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion, x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf = model(x_batch)
                    else:
                        auxiliary_atom, auxiliary_monomer, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion = model(x_batch)
                elif use_atom_model and use_monomer_model:
                    auxiliary_atom, auxiliary_monomer, auxiliary_output_atom, auxiliary_output_monomer, output_fusion = model(x_batch)
                elif use_atom_model and use_peptide_model:
                    auxiliary_atom, auxiliary_peptide, auxiliary_output_atom, auxiliary_output_peptide, output_fusion = model(x_batch)
                elif use_monomer_model and use_peptide_model:
                    auxiliary_monomer, auxiliary_peptide, auxiliary_output_monomer, auxiliary_output_peptide, output_fusion = model(x_batch)
            else:
                pred__ = model(x_batch)

            # During training, calculate validation loss for early stopping
            if istrain:
                if use_auxiliary:
                    if use_atom_model:
                        loss_atom = criterion(auxiliary_atom, logPapp__)
                        loss_output_atom = criterion(auxiliary_output_atom, logPapp__)
                    else:
                        loss_atom = 0.0
                        loss_output_atom = 0.0

                    if use_monomer_model:
                        loss_monomer = criterion(auxiliary_monomer, logPapp__)
                        loss_output_monomer = criterion(auxiliary_output_monomer, logPapp__)
                    else:
                        loss_monomer = 0.0
                        loss_output_monomer = 0.0

                    if use_peptide_model:
                        loss_peptide = criterion(auxiliary_peptide, logPapp__)
                        loss_output_peptide = criterion(auxiliary_output_peptide, logPapp__)
                    else:
                        loss_peptide = 0.0
                        loss_output_peptide = 0.0

                    loss_output_fusion = criterion(output_fusion, logPapp__)

                    if ensemble:
                        loss__ = gamma_layer*loss_atom + gamma_layer*loss_monomer + gamma_layer*loss_peptide + loss_output_fusion
                    else:
                        loss__ = gamma_layer*loss_atom + gamma_layer*loss_monomer + gamma_layer*loss_peptide + gamma_subout*loss_output_atom + gamma_subout*loss_output_monomer + gamma_subout*loss_output_peptide + loss_output_fusion
                else:
                    loss__ = criterion(pred__, logPapp__)

                running_loss += loss__.item()

            # During validation, calculate predictions for evaluation
            else:
                if use_auxiliary:
                    pred__ = output_fusion
                pred__ = pred__.detach().cpu()
                # IMPORTANT: scalling the predicted permeability values
                if not classification:
                    pred__ = torch.min(torch.max(pred__, torch.Tensor([LOWER_LIMIT]).repeat(pred__.shape)), torch.Tensor([UPPER_LIMIT]).repeat(pred__.shape))

                ids += id__.flatten().tolist()
                exps += logPapp__.flatten().tolist()
                preds += pred__.clone().numpy().tolist()

                # TODO: x_attn_graph, x_attn_conf, attn_weights_graph, attn_weights_conf can be used for attention visualization
                if visualization:
                    x_attn_graphs.append(x_attn_graph.detach().cpu().clone().numpy())
                    x_attn_confs.append(x_attn_conf.detach().cpu().clone().numpy())
                    attn_weights_graphs.append(attn_weights_graph.detach().cpu().clone().numpy())
                    attn_weights_confs.append(attn_weights_conf.detach().cpu().clone().numpy())

        if istrain:
            running_loss = running_loss / len(dataloader_valid)
            return running_loss
        else:
            if visualization:
                return ids, exps, preds, x_attn_graphs, x_attn_confs, attn_weights_graphs, attn_weights_confs
            else:
                return ids, exps, preds




def train_loop(model_path, device, patience, epoch_num,
               dataloader_train, dataloader_valid, model, criterion,
               optimizer, scheduler,
               # For early stopping
               use_earlystopping=True, earlystopping_type='Valid', verbose=False,
               use_auxiliary=False, gamma_layer=None, gamma_subout=None,
               use_atom_model=True, use_monomer_model=True, use_peptide_model=True, ensemble=False,
               visualization=False,
               classification=False,):
    """
    Training loop.
    """
    if use_earlystopping:
        earlystopping = EarlyStopping(patience=patience, path=model_path)

    loss_train_list, loss_valid_list = [], []

    for epoch in range(epoch_num):

        time_start = time.time()

        loss_train = train_epoch(device, epoch, model, dataloader_train, optimizer, criterion, scheduler, verbose=verbose, \
                                 use_auxiliary=use_auxiliary, gamma_layer=gamma_layer, gamma_subout=gamma_subout, \
                                 use_atom_model=use_atom_model, use_monomer_model=use_monomer_model, use_peptide_model=use_peptide_model, \
                                 ensemble=ensemble, \
                                 visualization=visualization, \
                                 classification=classification)
        loss_train_list.append(loss_train)
        print(f'Train loss of epoch {epoch:>2d}: {loss_train:.6f}')

        loss_valid = predict_valid(device, model, dataloader_valid, criterion, use_auxiliary=use_auxiliary, gamma_layer=gamma_layer, gamma_subout=gamma_subout, \
                                   use_atom_model=use_atom_model, use_monomer_model=use_monomer_model, use_peptide_model=use_peptide_model, \
                                   ensemble=ensemble, \
                                   visualization=visualization,\
                                   classification=classification)
        loss_valid_list.append(loss_valid)
        print(f'Valid loss of epoch {epoch:>2d}: {loss_valid:.6f}')

        time_end = time.time()
        print(f'Time of epoch {epoch:>2d}: {(time_end-time_start):.3f}')

        if use_earlystopping:
            if earlystopping_type == 'Valid':
                earlystopping(loss_valid, epoch, model, optimizer, loss_train_list, loss_valid_list)
                if earlystopping.early_stop:
                    print("Early Stopping!!!!!")
                    break
            # WARNING: Using training loss for early stopping is not recommended
            elif earlystopping_type == 'Train':
                earlystopping(loss_train, epoch, model, optimizer, loss_train_list, loss_valid_list)
                if earlystopping.early_stop:
                    print("Early Stopping!!!!!")
                    break

        print("------------------------------------")


    if not use_earlystopping:
        print(f'Saving model ...')
        # save model
        torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_now': loss_valid,
                    'loss_train_list': loss_train_list,
                    'loss_valid_list': loss_valid_list,
                    }, model_path)
        print("------------------------------------------------------------------------")
    else:
        print("------------------------------------------------------------------------")
        return loss_train_list, loss_valid_list




def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def evaluate_model(X, Y, round_num=3, classification=False, preds_probs=None):
    """
    Evaluate prediction performance.
    return:
        acc, precision, recall, f1, auc, mcc for classification task
        MAE, RMSE, R, MSE, R2 for regression task
    """
    if len(X) < 2:
        return 'nan', 'nan'
    else:
        if classification:
            acc = np.round(accuracy_score(X, Y), round_num)
            precision = np.round(precision_score(X, Y), round_num)
            recall = np.round(recall_score(X, Y), round_num)
            f1 = np.round(f1_score(X, Y), round_num)
            auc = np.round(roc_auc_score(X, preds_probs), round_num)
            mcc = np.round(matthews_corrcoef(X, Y), round_num)
            return acc, precision, recall, f1, auc, mcc
        else:
            MAE = np.round(mean_absolute_error(X, Y), round_num)
            RMSE = np.round(np.sqrt(mean_squared_error(X, Y)), round_num)
            R = np.round(pearsonr(X, Y)[0], round_num)
            MSE = np.round(mean_squared_error(X, Y), round_num)
            R2 = np.round(r2_score(X, Y), round_num)
            return MAE, RMSE, R, MSE, R2



def load_dataset_cv(model_type, replica_num, cv):
    """
    Rearrange input data for different trials of cross-validation.
    TODO: There is room for improvement.
    """
    load_node_train = np.load(f'../input/Trans/{replica_num}/node_{replica_num}_Train.npz')
    load_node_valid = np.load(f'../input/Trans/{replica_num}/node_{replica_num}_Valid.npz')
    load_graph_train = np.load(f'../input/Trans/{replica_num}/graph_{replica_num}_Train.npz')
    load_graph_valid = np.load(f'../input/Trans/{replica_num}/graph_{replica_num}_Valid.npz')
    load_conf_train = np.load(f'../input/Trans/{replica_num}/conf_{replica_num}_Train.npz')
    load_conf_valid = np.load(f'../input/Trans/{replica_num}/conf_{replica_num}_Valid.npz')
    load_bond_train = np.load(f'../input/Trans/{replica_num}/bond_{replica_num}_Train.npz')
    load_bond_valid = np.load(f'../input/Trans/{replica_num}/bond_{replica_num}_Valid.npz')
    load_monomers_train = np.load(f'../input/CNN/{replica_num}/feature_map_{replica_num}_Train.npz')
    load_monomers_valid = np.load(f'../input/CNN/{replica_num}/feature_map_{replica_num}_Valid.npz')
    load_peptides_train = np.load(f'../input/MLP/{replica_num}/peptide_{replica_num}_Train.npz')
    load_peptides_valid = np.load(f'../input/MLP/{replica_num}/peptide_{replica_num}_Valid.npz')


    train_ids = np.load(f'data/eval_index/train_ids_cv{cv}.npy')
    valid_ids = np.load(f'data/eval_index/valid_ids_cv{cv}.npy')
    id = load_peptides_train['id'].tolist()+load_peptides_valid['id'].tolist()
    train_index, valid_index = [], []
    for i in range(len(id)):
        if id[i] in train_ids:
            train_index.append(i)
        elif id[i] in valid_ids:
            valid_index.append(i)
    train_index = sorted(train_index)
    valid_index = sorted(valid_index)


    id = np.vstack([load_peptides_train['id'].reshape(-1,1), load_peptides_valid['id'].reshape(-1,1)])
    y = np.vstack([load_peptides_train['y'].reshape(-1,1), load_peptides_valid['y'].reshape(-1,1)])


    if model_type == 'Trans':
        atoms_mask = np.vstack([load_node_train['atoms_mask'], load_node_valid['atoms_mask']])
        atoms_features = np.vstack([load_node_train['atoms_features'], load_node_valid['atoms_features']])
        graph = np.vstack([load_graph_train['graph'], load_graph_valid['graph']])
        conf = np.vstack([load_conf_train['conf'], load_conf_valid['conf']])
        bond = np.vstack([load_bond_train['bond'], load_bond_valid['bond']])

        dataset_train = DataSet(list(zip(torch.Tensor(id[train_index]).to(torch.int32),
                                        torch.Tensor(atoms_mask[train_index]).to(torch.float32),
                                        torch.Tensor(atoms_features[train_index]).to(torch.float32),
                                        torch.Tensor(graph[train_index]).to(torch.float32),
                                        torch.Tensor(conf[train_index]).to(torch.float32),
                                        torch.Tensor(bond[train_index]).to(torch.float32),
                                        torch.Tensor(y[train_index]).to(torch.float32))))
        dataset_valid = DataSet(list(zip(torch.Tensor(id[valid_index]).to(torch.int32),
                                        torch.Tensor(atoms_mask[valid_index]).to(torch.float32),
                                        torch.Tensor(atoms_features[valid_index]).to(torch.float32),
                                        torch.Tensor(graph[valid_index]).to(torch.float32),
                                        torch.Tensor(conf[valid_index]).to(torch.float32),
                                        torch.Tensor(bond[valid_index]).to(torch.float32),
                                        torch.Tensor(y[valid_index]).to(torch.float32))))

    elif model_type == 'CNN':
        table = np.vstack([load_monomers_train['table'], load_monomers_valid['table']])
        feature_map = np.vstack([load_monomers_train['feature_map'], load_monomers_valid['feature_map']])

        dataset_train = DataSet(list(zip(torch.Tensor(id[train_index]).to(torch.int32),
                                        torch.Tensor(table[train_index]).to(torch.int32),
                                        torch.Tensor(feature_map[train_index]).to(torch.float32),
                                        torch.Tensor(y[train_index]).to(torch.float32))))
        dataset_valid = DataSet(list(zip(torch.Tensor(id[valid_index]).to(torch.int32),
                                        torch.Tensor(table[valid_index]).to(torch.int32),
                                        torch.Tensor(feature_map[valid_index]).to(torch.float32),
                                        torch.Tensor(y[valid_index]).to(torch.float32))))

    elif model_type == 'MLP':
        peptide_descriptor = np.vstack([load_peptides_train['peptide_descriptor'], load_peptides_valid['peptide_descriptor']])
        fps = np.vstack([load_peptides_train['fps'], load_peptides_valid['fps']])

        dataset_train = DataSet(list(zip(torch.Tensor(id[train_index]).to(torch.int32),
                                        torch.Tensor(peptide_descriptor[train_index]).to(torch.float32),
                                        torch.Tensor(fps[train_index]).to(torch.float32),
                                        torch.Tensor(y[train_index]).to(torch.float32))))
        dataset_valid = DataSet(list(zip(torch.Tensor(id[valid_index]).to(torch.int32),
                                        torch.Tensor(peptide_descriptor[valid_index]).to(torch.float32),
                                        torch.Tensor(fps[valid_index]).to(torch.float32),
                                        torch.Tensor(y[valid_index]).to(torch.float32))))

    elif model_type == 'Fusion':
        atoms_mask = np.vstack([load_node_train['atoms_mask'], load_node_valid['atoms_mask']])
        atoms_features = np.vstack([load_node_train['atoms_features'], load_node_valid['atoms_features']])
        graph = np.vstack([load_graph_train['graph'], load_graph_valid['graph']])
        conf = np.vstack([load_conf_train['conf'], load_conf_valid['conf']])
        bond = np.vstack([load_bond_train['bond'], load_bond_valid['bond']])

        table = np.vstack([load_monomers_train['table'], load_monomers_valid['table']])
        feature_map = np.vstack([load_monomers_train['feature_map'], load_monomers_valid['feature_map']])

        peptide_descriptor = np.vstack([load_peptides_train['peptide_descriptor'], load_peptides_valid['peptide_descriptor']])
        fps = np.vstack([load_peptides_train['fps'], load_peptides_valid['fps']])

        dataset_train = DataSet(list(zip(torch.Tensor(id[train_index]).to(torch.int32),
                                        torch.Tensor(atoms_mask[train_index]).to(torch.float32),
                                        torch.Tensor(atoms_features[train_index]).to(torch.float32),
                                        torch.Tensor(graph[train_index]).to(torch.float32),
                                        torch.Tensor(conf[train_index]).to(torch.float32),
                                        torch.Tensor(bond[train_index]).to(torch.float32),
                                        torch.Tensor(table[train_index]).to(torch.int32),
                                        torch.Tensor(feature_map[train_index]).to(torch.float32),
                                        torch.Tensor(peptide_descriptor[train_index]).to(torch.float32),
                                        torch.Tensor(fps[train_index]).to(torch.float32),
                                        torch.Tensor(y[train_index]).to(torch.float32))))

        dataset_valid = DataSet(list(zip(torch.Tensor(id[valid_index]).to(torch.int32),
                                        torch.Tensor(atoms_mask[valid_index]).to(torch.float32),
                                        torch.Tensor(atoms_features[valid_index]).to(torch.float32),
                                        torch.Tensor(graph[valid_index]).to(torch.float32),
                                        torch.Tensor(conf[valid_index]).to(torch.float32),
                                        torch.Tensor(bond[valid_index]).to(torch.float32),
                                        torch.Tensor(table[valid_index]).to(torch.int32),
                                        torch.Tensor(feature_map[valid_index]).to(torch.float32),
                                        torch.Tensor(peptide_descriptor[valid_index]).to(torch.float32),
                                        torch.Tensor(fps[valid_index]).to(torch.float32),
                                        torch.Tensor(y[valid_index]).to(torch.float32))))

    return dataset_train, dataset_valid