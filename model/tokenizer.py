import re

class Tokenizer:
    def __init__(
        self,
        token_set,
        tokenized_smiles=None):
        self.token_set = token_set
        self.token_set.insert(0, self.eos)
        self.token_set.insert(0, self.bos)
        self.token_set.insert(0, self.pad)
        
        self.tokenized_smiles = tokenized_smiles
    
    @classmethod
    def from_smiles(
        cls,
        smiles_list):
        tokenized_smiles_list = []
        unique_token_set = set()
        for smi in smiles_list:
            tokenized_smiles = cls.tokenize_smiles(smi)
            unique_token_set |= set(tokenized_smiles)
            tokenized_smiles_list.append(tokenized_smiles)
        return cls(sorted(list(unique_token_set)), tokenized_smiles_list)
    
    @classmethod
    def from_file(
        cls,
        vocab_file):
        with open(vocab_file, "r") as f:
            token_set = f.readlines()[3:]
        token_set = [t.strip('\n') for t in token_set]
        return cls(token_set)
    
    @property
    def bos(self):
        return '<bos>'
    
    @property
    def eos(self):
        return '<eos>'
    
    @property
    def pad(self):
        return '<pad>'
    
    @property
    def bos_id(self):
        return self.token_set.index('<bos>')
    
    @property
    def eos_id(self):
        return self.token_set.index('<eos>')
    
    @property
    def pad_id(self):
        return self.token_set.index('<pad>')
    
    @property
    def vocab(self):
        return self.token_set
    
    @property
    def data_size(self):
        return len(self.tokenized_smiles)
    
    @staticmethod
    def tokenize_smiles(smiles):
        """
        This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
        Modified by Shoichi Ishida
        """
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        assert smiles == ''.join(tokens)
        return tokens
    
    def prepare_data(self):
        X, y = [], []
        for ts in self.tokenized_smiles:
            X.append(self.tok2int(ts, add_bos=True, add_eos=False))
            y.append(self.tok2int(ts, add_bos=False, add_eos=True))
        return X, y
    
    def smi2int(self, smiles, add_bos=False, add_eos=False):
        tokens = self.tokenize_smiles(smiles)
        if add_bos:
            tokens.insert(0, self.bos)
        if add_eos:
            tokens.append(self.eos)
        ints = self.tok2int(tokens)
        return ints
    
    def tok2int(self, tokens, add_bos=False, add_eos=False):
        ints = [self.token_set.index(token) for token in tokens]
        if add_bos:
            ints.insert(0, self.token_set.index(self.bos))
        if add_eos:
            ints.append(self.token_set.index(self.eos))
        return ints
    
    def int2smi(self, ints, remove_bos=False, remove_eos=False):
        if remove_bos:
            bos_pos = ints.index(self.bos_id)
            ints = ints[bos_pos+1:]
        if remove_eos:
            eos_pos = ints.index(self.eos_id)
            ints = ints[:eos_pos]
        smi = "".join([self.token_set[i] for i in ints])
        return smi
    
    def save_vocab(self, path_file):
        with open(path_file, "w") as f:
            for v in self.vocab:
                f.write(f"{v}\n")