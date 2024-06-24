"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 6 Jan, 2024
"""

import albumentations as A
import datetime

import numpy as np
from pprint import pprint
import random
import torch 

import vocabulary as vocab
import utils

# I/O
SEED = 2023
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
# Paths
DATASET_PATH = "../../../dataset/robot_manipulations/"
MODEL_PATH = "../models/"
LOGS_PATH = "../logs/"
LOGGING_FILE = f"../logs/logs_{formatted_datetime}.txt"
MODEL_LOGGING_FILE = "../logs/model_config.txt"

# Vocabulary & Maps
SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]"]
SPECIAL_TOKEN_IDS = [0, 1, 2]
TARGETS         = SPECIAL_TOKENS + [tok for tok in vocab.OBJECTS+vocab.MOTOR_COMMANDS]
TARGETS_MAPPING = {tok:idx for idx,tok in enumerate(TARGETS)}
TARGETS_REVERSE_MAPPING = {idx:tok for idx,tok in enumerate(TARGETS)}
TARGET_VOCAB_SIZE = len(TARGETS)

# Special tokens
SRC_PAD_TOK_ID = 0
TGT_PAD_TOK_ID = TARGETS_MAPPING["[PAD]"]

EOS_TOKEN_ID = TARGETS_MAPPING["[EOS]"]
SOS_TOKEN_ID = TARGETS_MAPPING["[SOS]"]

SOS_TOKEN = TARGETS_REVERSE_MAPPING[1]
EOS_TOKEN = TARGETS_REVERSE_MAPPING[2]

# Inputs & Tokenizer
# Constants for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 288 # default size for efficientnet B3: will produce a feature map of shape (B, EMBEDDING_DIM, 9, 9)
IMG_RESIZE = 384

TRAIN_TFMS        = {
    "Resize"                  : {"height": IMG_SIZE,"width": IMG_SIZE},
    "RandomBrightnessContrast": {'p': 0.35},
}
TEST_TFMS        = {
    "Resize"                  : {"height": IMG_SIZE,"width": IMG_SIZE},
}  
NUM_HISTORY = 5
HISTORY_AUGS = A.Compose([
    A.Defocus(),
    A.Emboss(p=0.5),
    A.Perspective(),
    A.CoarseDropout(p=0.4),
    A.Sharpen()
])
MAX_LEN = 16
NUM_DECODER_INP_IDS_FOR_LEARNED_TOKENS = 2
VALIDATION_PCT = .1

# model
IMG_ENCODER_BACKBONES = {
    "resnet18"        : "resnet18.a1_in1k", # 11.7M
    "resnet34"        : "resnet34.a1_in1k", # 21.8M
    "resnet50"        : "resnet50.a1_in1k", #23.5M
    "convnext_nano"   : "convnextv2_nano.fcmae_ft_in22k_in1k", # 15.6M
    "convnext_tiny"   : "convnextv2_tiny.fcmae_ft_in22k_in1k", # 28.6M
    "efficientnet_b3" : "efficientnet_b3.ra2_in1k", # 10M
    "efficientnet_b4" : "efficientnet_b4.ra2_in1k", # 19.3M
    "mobilenet-v3-small": "mobilenetv3_small_100.lamb_in1k", # 2M
    "mobilenet-v3-large": "mobilenetv3_large_100.ra_in1k", # 5.5M,
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k", # 5.7M
}

SELECTED_CNN_BACKBONE = "efficientnet_b3"
FREEZE_CNN = True
DATA_SUBSET = False
LANG_MODEL_NAME = 'prajjwal1/bert-small'
TOKENIZER_CONFIG = {
    "do_lower_case": True
}
WANDB_LOGGING  = True

# training
RUN_NAME = "be_model"
GROUP_NAME = "RT1-CRAM"
PROJECT_NAME = 'SMF-Be'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DEVICE = "cpu"
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1
EPOCHS = 100
NUM_VAL_STEPS = 1

LR = 3e-3
LR_SCHEDULE_START = 200
STEPS_PER_EPOCH = 10
MAX_LR = LR
LR_START = 1e-5
LR_MAX = 3e-4
LR_MIN = 1e-7
LR_EXP_DECAY = .99

OPTIMIZER = "Adam"
NUM_WORKERS = 1 #4
LABEL_SMOOTHING = 0.0
GRAD_CLIP_VAL = 2.
WEIGHT_DECAY = 2e-6
REDUCE_LR_SCHEDULER = {
    "type": "ReduceLROnPlateau",
    "params": {
        "mode":'min', 
        "factor":0.2, 
        "patience":10, 
        "min_lr":1e-8, 
        "verbose":  False
    }
}
CYCLYC_LR_SCHEDULER = {
    "type": "OneCycleLR",
    "params": {
        "max_lr": MAX_LR,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs":EPOCHS // STEPS_PER_EPOCH,
        "anneal_strategy":"cos"
    }
}

WARMUP_LR_SCHEDULER = {
    "type": "LambdaLR",
    "params": {
        "lr_lambda" : lambda epoch: utils.model_utils.lrfn(epoch)
    }
}
 # select LR scheduler

LR_SCHEDULER  = WARMUP_LR_SCHEDULER
# update LR for warmup lr scheduling
LR = 1. if LR_SCHEDULER["type"] == "LambdaLR" else LR

## Robotics Transformer
### Encoder
NUM_RES_BLOCKS = 6
NUM_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "convnext_nano": 640,
    "convnext_tiny": 768,
    "efficientnet_b3": 384,
    "efficientnet_b4": 448,
    "resnet50": 2048,
}
ENCODER_DROPOUT_RATE = 0.1
EMBEDDING_DIM = 512
DIM_VL_TOKENS = EMBEDDING_DIM

TOKEN_LEARNER_DROPOUT = 0.1
TOKEN_LEARNER_DIM = 256

### Decoder
INF = float("-inf") # -1e9
IMG_TOKEN_SIZE = 7
NUM_LEARNED_TOKENS = 8
NUM_TOKENIZED_INPUTS = (1+NUM_HISTORY)*NUM_LEARNED_TOKENS
N_DECODER_LAYERS = 2
N_HEADS = 8
EXPANSION = 2
D_MODEL = EMBEDDING_DIM
D_K = D_MODEL // N_HEADS 
D_FF = 1024
DECODER_DROPOUT_RATE = 0.1
ACTION_BINS = 256
MAX_OUT_SEQ_LEN = 16
NUM_ACTION_SLOTS = 9 # discrete action space as in RT1 

if __name__ == "__main__":
    pprint(TARGETS[:4])
    pprint(TARGETS_MAPPING)
