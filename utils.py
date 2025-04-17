import torch, random
import torch.nn as nn
import numpy as np
import logging
from torch_geometric.data import DataLoader
from Dataset import MolNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class metrics_c(nn.Module):
    def __init__(self, acc_f, pre_f, rec_f,auc_f):
        super(metrics_c, self).__init__()
        self.acc_f = acc_f
        self.pre_f = pre_f
        self.rec_f = rec_f
        self.auc_f = auc_f

    def forward(self, out, prob, tar):
        if len(out.shape) > 1:
            acc, f1, pre, rec, auc = [], [], [], [], []
            
            for i in range(out.shape[-1]):
                acc_, f1_, pre_, rec_, auc_ = 0, 0, 0, 0, 0
                acc_ = self.acc_f(tar[:, i], out[:, i])
                pre_ = self.pre_f(tar[:, i], out[:, i])
                rec_ = self.rec_f(tar[:, i], out[:, i])
                auc_ = self.auc_f(tar[:, i], prob[:, i])
                
                acc.append(acc_);  pre.append(pre_); rec.append(rec_); auc.append(auc_)
            return acc, f1, pre, rec, auc
        else:
            acc = self.acc_f(tar, out)
            pre = self.pre_f(tar, out)
            rec = self.rec_f(tar, out)
            auc = self.auc_f(tar, prob)
        return acc, pre, rec, auc

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

import random

def shuffle_data(data, seed=None):
    """
    Shuffle data with an optional seed to ensure reproducibility.

    Parameters:
    - data: The dataset object that supports indexing (e.g., slicing).
    - seed: Optional; an integer seed for reproducibility.

    Returns:
    - shuffled_data: A shuffled version of the input data.
    """
    # Convert data to a list of indices
    indices = list(range(len(data)))

    # Set the random seed
    if seed is not None:
        random.seed(seed)
    
    # Shuffle the indices
    random.shuffle(indices)

    # Reorder the data based on the shuffled indices
    shuffled_data = [data[i] for i in indices]

    return shuffled_data

def load_data(dataset_name, shuffle = True):
    data = MolNet(root='./dataset', dataset=dataset_name)
    if shuffle == True:
        data = shuffle_data(data, seed=42)
    return data


def load_fold_data(i, batch, cpus_per_gpu, k, dataset):

    data = dataset
    folds = k + 1
    fold_size = len(data) // folds
    test_start = len(data) - fold_size
    val_start = i * fold_size
    test_set = data[test_start:]
    if i != k - 1:
                val_end = (i + 1) * fold_size
                valid_set = data[val_start:val_end]
                train_set = data[0:val_start] + data[val_end:test_start]
    else:
                valid_set = data[val_start:test_start]
                train_set = data[0:val_start]

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=False, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)

    return train_loader, valid_loader, test_loader  


def create_ffn(task, tasks, output_dim, dropout):
    if task == 'clas':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks),
            nn.Sigmoid())
    elif task == 'reg':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks))
    else:
        raise NameError('task must be reg or clas!')
    return act


def get_attn_pad_mask(mask): 
    batch_size, len_q = mask.size(0), mask.size(1)
    a = mask.unsqueeze(1).expand(batch_size, len_q, len_q)
    pad_attn_mask = a * a.transpose(-1, -2) 
    return pad_attn_mask.data.eq(0) 
