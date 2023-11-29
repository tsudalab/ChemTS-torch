import torch
from torch.nn.utils.rnn import pad_sequence

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    
class SmilesCondDataset(SmilesDataset):
    def __init__(self, substruct, mol_X, mol_y):
        super().__init__(mol_X, mol_y)
        self.substruct = substruct
        
    def __getitem__(self, idx):
        return self.substruct[idx], self.X_train[idx], self.y_train[idx]

def collate_right_pad(batch):
    X = [torch.tensor(b[0]) for b in batch]
    y = [torch.tensor(b[1]) for b in batch]
    
    seq_lengths = torch.tensor([len(x) for x in X], dtype=torch.int64)  # pylint: disable=not-callable
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    return X, y, seq_lengths


def collate_cond(batch, conf):
    seq_lengths = None
    sub_len = conf["Data"]["sub_len"]
    seq_len = conf["Data"]["seq_len"]
    
    sub = []
    X = []
    y = []
    for b in batch:
        sub.append(b[0] + [0] * (sub_len - len(b[0])))
        X.append(b[1] + [0] * (seq_len - len(b[1])))
        y.append(b[2] + [0] * (seq_len - len(b[2])))
        
    
    sub = torch.tensor(sub)
    X = torch.tensor(X)
    y = torch.tensor(y)
    return (sub, X), y, seq_lengths

def collate_left_pad(batch, max_len):
    X = []
    y = []
    for b in batch:
        X.append([0 for _ in range(max_len - len(b[0]))] + b[0])
        y.append([0 for _ in range(max_len - len(b[1]))] + b[1])
    
    X = torch.tensor(X)
    y = torch.tensor(y)
    
    return X, y, None
    