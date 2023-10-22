import argparse
import os
import sys

import numpy as np
import yaml

from chemts.preprocessing import read_smiles_dataset
from model.dataset import SmilesDataset, collate_fn
from model.model import StringPredModule
from model.tokenizer import SmilesTokenizer, SelfiesTokenizer

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    return parser.parse_args()

def main():
    args = get_parser()
    # Setup configuration
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"========== Configuration ==========")
    for k, v in conf.items():
        print(f"{k}: {v}")
    print(f"===================================")
    
    # Set seeds
    np.random.seed(conf['Seed'])
    torch.manual_seed(conf['Seed'])
    torch.cuda.manual_seed(conf['Seed'])

    os.makedirs(conf['Data']['output_model_dir'], exist_ok=True)

    # Prepare training dataset
    original_smiles = read_smiles_dataset(conf['Data']["dataset"])
    if conf["Data"]["format"].lower() == "smiles":
        Tokenizer = SmilesTokenizer
    elif conf["Data"]["format"].lower() == "selfies":
        Tokenizer = SelfiesTokenizer
    else:
        raise ValueError(f'Data format should be "smiles" or "selfies", but got "{conf["Data"]["format"]}"!')
    tokenizer = Tokenizer.from_smiles(original_smiles)
    tokenizer.save_tokens(conf['Data']['output_token'])
    print(f"[INFO] Save generated tokens to {conf['Data']['output_token']}")
    
    X, y = tokenizer.prepare_data(original_smiles)

    print(f"vocabulary:\n{tokenizer.tokens}\n"
          f"size of SMILES list: {len(X)}")
    
    conf["Data"]["vocab_len"] = len(tokenizer.token_set)
    
    with open(args.config, 'w') as file:
        yaml.dump(conf, file)
    print("Config files updated.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=conf['Train']['validation_split'], random_state=conf['Seed'])

    dataset_train = SmilesDataset(X_train, y_train)
    dataloader_train = DataLoader(
        dataset_train, batch_size=conf['Train']['batch_size'], shuffle=True,
        num_workers=conf['Train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    dataset_val = SmilesDataset(X_val, y_val)
    dataloader_val = DataLoader(
        dataset_val, batch_size=conf['Train']['batch_size'], shuffle=False, 
        num_workers=conf['Train']['num_workers'], pin_memory=True, collate_fn=collate_fn)

    # Build model
    
    model = StringPredModule(conf)
    
    logger = CSVLogger("logs", name="smiles_rnn_training")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        dirpath=conf['Data']['output_model_dir'],
        filename="smi_acc-{epoch:02d}-{val_acc:.2f}")
    
    earlystopping_callback = EarlyStopping("val_acc", mode="max", 
        patience=conf["Train"]["patience"])
    
    model_summary_callback = ModelSummary(max_depth=1)
        
    trainer = pl.Trainer(
        max_epochs=conf["Train"]["epoch"], 
        accelerator=conf["Train"]["accelerator"], 
        logger=logger,  
        devices=[conf["Train"]["device"]], 
        callbacks=[checkpoint_callback, earlystopping_callback, model_summary_callback],
        # gradient_clip_val=conf["Train"]["gradient_clip"])
    )    
    trainer.fit(
        model=model, 
        train_dataloaders=dataloader_train, 
        val_dataloaders=dataloader_val)
    
    print(f"[INFO] Save a training log to {conf['Data']['output_model_dir']}")


if __name__ == "__main__":
    main()
