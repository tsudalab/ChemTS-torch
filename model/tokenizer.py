import re
from abc import ABC, abstractmethod
import selfies

class BaseTokenizer(ABC):
    def __init__(
        self,
        token_set):
        self.token_set = token_set        
    
    @classmethod
    @abstractmethod
    def from_smiles(cls):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def tokenize_string(self, string):
        raise NotImplementedError()
    
    @classmethod
    def from_file(
        cls,
        vocab_file):
        with open(vocab_file, "r") as f:
            token_set = f.readlines()
        token_set = [t.strip('\n') for t in token_set]
        return cls(token_set)
    
    @property
    def tokens(self):
        return self.token_set
    
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
    def tokens(self):
        return self.token_set
    
    def prepare_data(self, smiles_list):
        tokenized_string_list = []
        for smi in smiles_list:
            tokenized_smiles = self.tokenize_string(smi)
            tokenized_string_list.append(tokenized_smiles)
        X, y = [], []
        for ts in tokenized_string_list:
            X.append(self.tokens_to_ints(ts, add_bos=True, add_eos=False))
            y.append(self.tokens_to_ints(ts, add_bos=False, add_eos=True))
        return X, y
    
    def string_to_ints(self, string, add_bos=False, add_eos=False):
        tokens = self.tokenize_string(string)
        if add_bos:
            tokens.insert(0, self.bos)
        if add_eos:
            tokens.append(self.eos)
        ints = self.tokens_to_ints(tokens)
        return ints
    
    def tokens_to_ints(self, tokens, add_bos=False, add_eos=False):
        ints = [self.token_set.index(token) for token in tokens]
        if add_bos:
            ints.insert(0, self.token_set.index(self.bos))
        if add_eos:
            ints.append(self.token_set.index(self.eos))
        return ints
    
    def ints_to_tokens(self, ints, remove_bos=False, remove_eos=False):
        if remove_bos:
            bos_pos = ints.index(self.bos_id)
            ints = ints[bos_pos+1:]
        if remove_eos:
            eos_pos = ints.index(self.eos_id)
            ints = ints[:eos_pos]
        tokens = [self.token_set[i] for i in ints]
        return tokens
    
    def ints_to_string(self, ints, remove_bos=False, remove_eos=False):
        tokens = self.ints_to_tokens(ints, remove_bos, remove_eos)
        smi = "".join(tokens)
        return smi
    
    def save_tokens(self, path_file):
        with open(path_file, "w") as f:
            for v in self.token_set:
                f.write(f"{v}\n")
                
class SmilesTokenizer(BaseTokenizer):
    def __init__(self, token_set):
        super().__init__(token_set)
        
    @classmethod
    def from_smiles(
        cls,
        smiles_list):
        unique_token_set = set()
        for smi in smiles_list:
            tokenized_smiles = cls.tokenize_string(smi)
            unique_token_set |= set(tokenized_smiles)
        token_set = sorted(list(unique_token_set))
        token_set.insert(0, '<eos>')
        token_set.insert(0, '<bos>')
        token_set.insert(0, '<pad>')
        return cls(token_set)
    
    @ staticmethod
    def tokenize_string(smiles, add_bos=False, add_eos=False):
        """
        This function is based on https://github.com/pschwllr/MolecularTransformer#pre-processing
        Modified by Shoichi Ishida
        """
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        assert smiles == ''.join(tokens)
        if add_bos:
            tokens.insert(0, "<bos>")
        if add_eos:
            tokens.append("<eos>")
        return tokens
    
    def ints_to_smiles(self, ints, remove_bos=False, remove_eos=False):
        tokens = self.ints_to_tokens(ints, remove_bos, remove_eos)
        smi = "".join(tokens)
        return smi


class SelfiesTokenizer(BaseTokenizer):
    def __init__(self, token_set):
        super().__init__(token_set)
         
    @classmethod
    def from_smiles(
        cls,
        smiles_list):
        """Build a SelfiesTokenizer object by extracting tokens from a list of smiles

        Args:
            smiles_list (List): smiles to extract tokens

        Returns:
            SelfiesTokenizer
        """
        selfies_list = [selfies.encoder(smi) for smi in smiles_list]
        token_set = selfies.get_alphabet_from_selfies(selfies_list)
        
        token_set = sorted(list(token_set))
        token_set.insert(0, '<eos>')
        token_set.insert(0, '<bos>')
        token_set.insert(0, '<pad>')
        return cls(token_set)
    
    @staticmethod
    def tokenize_string(smiles, add_bos=False, add_eos=False):
        selfies_str = selfies.encoder(smiles)
        tokens = list(selfies.split_selfies(selfies_str))
        assert selfies_str == ''.join(tokens)
        if add_bos:
            tokens.insert(0, "<bos>")
        if add_eos:
            tokens.append("<eos>")
        return tokens
    
    def ints_to_smiles(self, ints, remove_bos=False, remove_eos=False):
        selfies_str = self.ints_to_string(ints, remove_bos, remove_eos)
        smi = selfies.decoder(selfies_str)
        return smi
    
    def smiles_to_selfies(smiles):
        selfies_str = selfies.encoder(smiles)
        return selfies_str
    
    def selfies_to_smiles(selfies_str):
        smiles = selfies.decoder(selfies_str)
        return smiles