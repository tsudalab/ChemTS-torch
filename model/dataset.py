import torch
from torch.nn.utils.rnn import pad_sequence

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X_train[idx]), \
            torch.tensor(self.y_train[idx])

def collate_fn(batch):
    X = [b[0] for b in batch]
    y = [b[1] for b in batch]
    
    seq_lengths = torch.tensor([len(x) for x in X], dtype=torch.int64)  # pylint: disable=not-callable
    X =  pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    return X, y, seq_lengths