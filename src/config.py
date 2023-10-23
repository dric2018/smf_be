"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 19 Oct, 2023
"""
import numpy as np
from pprint import pprint
import random
import torch 
import vocabulary as vocab

# I/O
## Set seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
DATASET_PATH = "../../../dataset/robot_manipulations/"
MODEL_PATH = "../../../models/be_model.pt"

# Vocabulary & Maps
TARGETS         = ["[SOS]", "[PAD]", "[EOS]"] + [tok for tok in vocab.OBJECTS+vocab.MOTOR_COMMANDS]
TARGETS_MAPPING = {tok:idx for idx,tok in enumerate(TARGETS)}
TARGETS_REVERSE_MAPPING = {idx:tok for idx,tok in enumerate(TARGETS)}

# Inputs & Tokenizer
# Constants for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TFMS        = {
    "Resize"                  : {"height": 224,"width": 224},
    "RandomBrightnessContrast": {'p': 0.2},
}
TEST_TFMS        = {
    "Resize"                  : {"height": 224,"width": 224},
}  
MAX_LEN = 16
VALIDATION_PCT = .26

# model
IMG_ENCODER_BACKBONES = {
    "resnet18"        : "ResNet18_Weights.IMAGENET1K_V1",
    "resnet34"        : "ResNet34_Weights.IMAGENET1K_V1",
    "resnet50"        : "ResNet50_Weights.IMAGENET1K_V1",
    "convnext_tiny"   : "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "efficientnet_b3" : "EfficientNet_B3_Weights.IMAGENET1K_V1",
    "efficientnet_b4" : "EfficientNet_B4_Weights.IMAGENET1K_V1"

}
LANG_MODEL_NAME = 'prajjwal1/bert-mini'
TOKENIZER_CONFIG = {
    "do_lower_case": True
}

# training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
OPTIMIZER = "AdamW" 
NUM_WORKERS = 4

## Transformer
DROPOUT_RATE = .2
### Encoder
TEXT_ENC_DROPOUT = 0.15
EMBEDDING_DIM = 256

### Decoder
NUM_IMG_TOKENS = 7
N_DECODER_LAYERS = 6
N_HEADS = 8
D_MODEL = 512
D_K = D_MODEL // N_HEADS
D_FF = 1024
TOKEN_LEARNER_FTRS_SHAPE = (BATCH_SIZE, NUM_IMG_TOKENS*NUM_IMG_TOKENS, D_MODEL)


if __name__ == "__main__":
    pprint(TARGETS[:4])
    pprint(TARGETS_MAPPING)
