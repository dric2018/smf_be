"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 6 Jan, 2024
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
import lightning.pytorch as pl
import logging

import numpy as np
import os
import os.path as osp

import pandas as pd 
from pprint import pprint
from PIL import Image

from sklearn.model_selection import train_test_split

import sys

from tokenizers import Tokenizer

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

logging.basicConfig(level="INFO")

class TargetEncoding:
    def __init__(self, inp:str=None) -> None:
        self.ids = None
        self.tokens = None
        self.attention_mask = None
        self.pad_id = config.TGT_PAD_TOK_ID

        if inp is not None:
            self._encode(inp)

    def tokenize_by_space(
        self, 
        full_string:str
    )->list:
        return full_string.split()
    
    def _encode(self, inp:str):
        dec_inp_tokens                  = self.tokenize_by_space(f"{config.SOS_TOKEN} "+inp)
        dec_inp_token_ids               = [config.TARGETS_MAPPING[t] for t in dec_inp_tokens]
        len_dec_inputs                  = len(dec_inp_token_ids)
        
        labels = self.tokenize_by_space(inp+f" {config.EOS_TOKEN}")
        label_token_ids               = [config.TARGETS_MAPPING[t] for t in labels]
        
        # compute padding length
        padding_len             = config.MAX_OUT_SEQ_LEN - len_dec_inputs
        self.dec_inp_token_ids  = dec_inp_token_ids + ([self.pad_id] * padding_len)
        self.label_token_ids    = label_token_ids + ([self.pad_id] * padding_len)
        
class InputPreprocessor:
    def __init__(self):
            self.tokenizer          = AutoTokenizer.from_pretrained(config.LANG_MODEL_NAME)
            self.included_special_tokens = [
                self.tokenizer.cls_token_id, 
                self.tokenizer.sep_token_id, 
                self.tokenizer.pad_token_id
            ]

            tfms = [
                getattr(A, tfms)(**params) for tfms, params in config.TEST_TFMS.items()
            ]
            tfms.append(ToTensorV2())
            
            self.img_tfms = A.Compose(tfms)

    def _preprocess_inputs(self, imgs, instruction)->dict:
        # apply transforms
        in_state    = self.img_tfms(image=imgs)["image"]
        
        # prepare action desc & encoder inputs 
        enc_ad = self.tokenizer.encode_plus(
            instruction.strip(), 
            padding="max_length", 
            max_length=config.MAX_LEN
        )
        
        encoder_inp = torch.as_tensor(enc_ad["input_ids"]).long()     
        
        sample = {
            "sample_id": None,
            "in_state": in_state,
            "action_desc": {
                "raw"       : instruction,
                "ids"       : encoder_inp,
                "mask"      : torch.as_tensor(enc_ad["attention_mask"]).long(),
                "token_type_ids": torch.as_tensor(enc_ad["token_type_ids"]).long(),
            },
            
        }

        return sample

class BEDataset(Dataset):
    def __init__(
            self,
            df:pd.DataFrame,
            apply_transforms: bool = True,
            task: str = "train"              
        ) -> None:
        super().__init__()

        self.tokenizer          = AutoTokenizer.from_pretrained(config.LANG_MODEL_NAME)
        self.included_special_tokens = [
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id, 
            self.tokenizer.pad_token_id
        ]

        self.dataset_directory  = config.DATASET_PATH
        self.df                 = df.copy()
        self.task               = task
        self.apply_transforms   = apply_transforms
        
        # image transforms
        self.apply_transforms = apply_transforms
        if task == "train":
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in config.TRAIN_TFMS.items()
            ]
        else:
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in config.TEST_TFMS.items()
            ]

        tfms.append(A.Normalize())
        tfms.append(ToTensorV2())
        self.transforms = A.Compose(tfms) if self.apply_transforms else None

    def __len__(self):
        """
            Get the number of samples in the dataset
        """
        return self.df.shape[0]
    
    def _decode_inputs(self, ids:list):
        """
            Convert action description token ids into word tokens
        """
        decoded_text = self.tokenizer.decode(ids)
        decoded_text = ' '.join([token for token in decoded_text.split() if token not in [self.tokenizer.cls_token, self.tokenizer.pad_token, self.tokenizer.sep_token]])
        
        return decoded_text
    
    def _decode_outputs(self, ids:list):
        """
            Convert motor commands token ids into word tokens
        """
        decoded_text = " ".join([config.TARGETS_REVERSE_MAPPING[idx] for idx in ids if idx not in [0, 1, 2]])        
        return decoded_text
    
    def _get_seq_len(self, ids):
        pass
    
    def __getitem__(self, idx, return_goal:bool=False):
        """
            Prepare and return a sample from the dataset when requested by the dataloader
        """
        data_point = self.df.iloc[idx]
        
        # visual inputs
        try:
            in_state_filename       = str(data_point.in_state)
            goal_state_filename       = str(data_point.goal_state)
            ## in state
            in_state = np.array(Image.open(osp.join(
                self.dataset_directory, 
                str(data_point.version), 
                str(data_point.sample_ID), 
                in_state_filename
            )))
            
            ## goal state
            goal_state = np.array(Image.open(osp.join(
                self.dataset_directory, 
                str(data_point.version), 
                str(data_point.sample_ID), 
                goal_state_filename
            )))

        except FileNotFoundError:
            in_state_filename       = str(data_point.in_state)+".png"
            goal_state_filename       = str(data_point.goal_state)+".png"

            ## in state
            in_state = np.array(Image.open(osp.join(
                self.dataset_directory, 
                str(data_point.version), 
                str(data_point.sample_ID), 
                in_state_filename
            )))
            
            ## goal state
            goal_state = np.array(Image.open(osp.join(
                self.dataset_directory, 
                str(data_point.version), 
                str(data_point.sample_ID), 
                goal_state_filename
            )))
            
        # apply image treansforms
        if self.apply_transforms:
            # apply transforms
            in_state = self.transforms(image=in_state)["image"]
            goal_state = self.transforms(image=goal_state)["image"]
            
        # Language inputs
        ## action desc
        enc_ad = self.tokenizer.encode_plus(
            data_point.action_description.strip(), 
            padding="max_length", 
            max_length=config.MAX_LEN
        )

        ## target
        enc_cmd = TargetEncoding(inp=data_point.motor_cmd)
        
        # prepare encoder inputs
        encoder_inp = torch.as_tensor(enc_ad["input_ids"]).long()     
        
        sample = {
            "sample_id": data_point.sample_ID,
            "in_state": in_state,
            "action_desc": {
                "raw"       : data_point.action_description,
                "ids"       : encoder_inp,
                "mask"      : torch.as_tensor(enc_ad["attention_mask"]).long(),
                "token_type_ids": torch.as_tensor(enc_ad["token_type_ids"]).long(),
            },
            
        }

        if self.task == "train" or self.task == "eval":
            dec_inp = torch.as_tensor(enc_cmd.dec_inp_token_ids).long()
            labels  = torch.as_tensor(enc_cmd.label_token_ids).long()
            
            # prepare targets
            sample.update({
                "motor_cmd": {
                    "raw"       : data_point.motor_cmd,
                    "decoder_inp_ids"       : dec_inp,
                    "labels":  labels, 
                },
            })
            
            if return_goal:
                sample.update({"goal_state": goal_state})

        return sample

class BEDataModule(pl.LightningDataModule):
    def __init__(
            self
        ) -> None:
        super().__init__()

        self.train_df = pd.read_csv(osp.join(config.DATASET_PATH, "train.csv"))
        self.test_df = pd.read_csv(osp.join(config.DATASET_PATH, "test.csv"))
        
    def setup(self, stage=None):
        """
            Defines all the operations to perform while building the datasets
        """
        # shuffle dataframe
        if config.DATA_SUBSET:
            # prepare 30% of the dataset for training
            self.train_df = self.train_df.sample(frac=0.3).reset_index(drop=True)
        else:
            self.train_df = self.train_df.sample(frac=1.).reset_index(drop=True)
        
        # train/val split
        random_indices = np.random.rand(len(self.train_df)) < (1-config.VALIDATION_PCT)

        train = self.train_df[random_indices].reset_index(drop=True)
        val = self.train_df[~random_indices].reset_index(drop=True)

        # train dataset
        self.train_ds = BEDataset(
            df=train,
            task='train'
        )
    
        training_data_size = len(self.train_ds)
        logging.info(
            f'Training on {training_data_size} samples.'
        )

        # validation dataset
        if len(val) > 0:
            self.val_ds = BEDataset(
                df=val,
                task='train'
            )
            validation_data_size = len(self.val_ds)

            logging.info(
                f'Validating on {validation_data_size} samples.'
            )
            
        #  test dataset
        self.test_ds = BEDataset(
                df=self.test_df,
                task='eval'
            )
        test_data_size = len(self.test_ds)

        logging.info(
            f'Testing on {test_data_size} samples.'
        )
                    
        print(f"Total # examples: {training_data_size+validation_data_size+test_data_size}")

    def train_dataloader(self):
        """
            Defines the structure of the training dataloader
        """
        return DataLoader(
            dataset=self.train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=config.NUM_WORKERS,
            drop_last=True
        )

    def val_dataloader(self):
        """
            Defines the structure of the validation dataloader
        """
        return DataLoader(
            dataset=self.val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self):
        """
            Defines the structure of the Test dataloader
        """
        return DataLoader(
            dataset=self.test_ds,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            drop_last=False
        )

if __name__ == "__main__":
    # os.system("nvidia-smi")

    # logging.info("Loading dataset...")
    # # reading data summary
    # csv = pd.read_csv(osp.join(config.DATASET_PATH, "dataset.csv"))

    # # building data object
    # ds = BEDataset(
    #     df=csv    
    # )

    # # fetching example
    # rand_idx = np.random.randint(low=0, high=len(ds))
    # ex = ds[rand_idx]

    # print("Dataset size: ", len(ds))
    # print("="*100)
    # print("ID: ", ex["sample_id"])
    # print(">> InState: ", ex["in_state"].shape)
    # print(">> GoalState: ", ex["goal_state"].shape)
    # print(">> Desc:")
    # pprint(ex["action_desc"])
    # print(">> Cmd:")
    # pprint(ex["motor_cmd"])
    # print("="*100)

    logging.info("Creating data module...")
    dm = BEDataModule()
    dm.setup()

    # test data loading I/O
#     print("="*100)
#     logging.info("\n>> train data loader")
#     print(f"# train batches\t: {len(dm.train_dataloader())}")
#     for data in dm.train_dataloader():
#         # pprint(data)
#         sample_id, in_state, ad, cmd = data["sample_id"], data["in_state"], data["action_desc"], data["motor_cmd"]
#         print("In \t\t\t: ", in_state.shape)
#         print("Action desc \t\t: ", ad["ids"].shape)
#         print("CMD \t\t\t: ", cmd["ids"].shape)
#         print("CMD(len) \t\t: ", cmd["length"].shape)
#         break

#     logging.info("\n\n>> val data loader")
#     print(f"# validation batches\t: {len(dm.val_dataloader())}")
#     for data in dm.val_dataloader():
#         # pprint(data)
#         sample_id, in_state, ad, cmd = data["sample_id"], data["in_state"], data["action_desc"], data["motor_cmd"]
#         print("In \t\t\t: ", in_state.shape)
#         print("Action desc \t\t: ", ad["ids"].shape)
#         print("CMD \t\t\t: ", cmd["ids"].shape)
#         break

#     print("="*100)
    
    b = next(iter(dm.val_dataloader()))
    print(b.keys())
