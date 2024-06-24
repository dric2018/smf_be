import os
import os.path as osp

import sys
sys.path.append("../src") # import all RTCRAM source code

import config

import dataloader

import logging
logging.basicConfig(level="INFO")

import utils

import numpy as np

from PIL import Image

from rt1 import RTCRAM

import streamlit as st 

from time import time

import torch
import torch.nn as nn
from torchinfo import summary


@st.cache_data
def init_session_states():
    st.session_state.COLUMNS            = ["sample_ID", "version", "action_description", "motor_cmd"]
    st.session_state.DATA_PATH          = "data/test.csv"
    st.session_state.IMG_DIR            = "../../data/"
    st.session_state.in_state           = None
    st.session_state.action_desctiption = None
    st.session_state.model              = None
    
def fetch_new_sample():

    in_state = None

    sample               = st.session_state.data.sample(n=1)
    sample_id            = str(sample.sample_ID.values[0])
    version              = sample.version.values[0]
    action_desctiption   = sample.action_description.values[0]

    sample_img_path      = os.path.join(st.session_state.IMG_DIR, version, sample_id, "0.PNG")
    try:
        in_state = Image.open(sample_img_path)
    except FileNotFoundError:
        logging.error(f"File {sample_img_path} does not exist...reload app to fetch another image")

    return in_state, action_desctiption

def load_sample():

    in_state, action_desctiption = fetch_new_sample()

    while in_state is None:
        in_state, action_desctiption = fetch_new_sample()
    
    return in_state, action_desctiption

def fetch_image(
        filename:str, 
        version:str, 
        ID:int
    )->np.ndarray:
    
    in_state_filename       = filename
    TEST_DATA_PATH               = ""

    try:
        
        img = np.array(Image.open(osp.join(
            TEST_DATA_PATH, 
            str(version), 
            str(ID), 
            in_state_filename
        )))

    except FileNotFoundError:
        in_state_filename       = filename+".png"

        img = np.array(Image.open(osp.join(
            TEST_DATA_PATH, 
            str(version), 
            str(ID), 
            in_state_filename
        )))
    
    return img

def preprocess_inputs(
        n_inputs:int=config.TEST_BATCH_SIZE,
        instruction:str=None,
        img:np.ndarray=None,        
        )->dict:
    
    """
        Preprocess/Create inputs to the RTCRAM model
    """

    if img is None:
        imgs = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3))
    else: 
        imgs = img

    if instruction is None:
        instruction = "shift the bottle right"

    # create dummy inputs
    preprocessor = dataloader.InputPreprocessor()

    sample = preprocessor._preprocess_inputs(imgs, instruction)    

    return sample

def test_model_io(
        instruction:str=None,
        img:np.ndarray=None,
        debug:bool=False, 
        device:str=config.DEVICE
    ):
    """
        Test RTCRAM I/O pipeline
    """
    st.info("Preparing inputs...")
    logging.info("Preparing inputs...")
    sample  = preprocess_inputs(instruction=instruction, img=img)
    in_state = sample["in_state"]
    raw_ad = sample["action_desc"]["raw"]
    ad_ids = sample["action_desc"]["ids"]

    # print("Input state shape: ", in_state.shape)
    # print("Raw instruction: ", raw_ad)
    # print("Input ids: ", ad_ids)

    # st.image(in_state.permute(1, 2, 0).numpy())
    # print(in_state.min(), in_state.max())

    batch_input = {
        "in_state"          : in_state.unsqueeze(0),
        "ids"               : ad_ids.unsqueeze(0),
        "mask"              : sample["action_desc"]["mask"].unsqueeze(0),
        "token_type_ids"    : sample["action_desc"]["token_type_ids"].unsqueeze(0)
    }    
    # st.write(f"`{summary(st.session_state.model)}`")
    
    st.info("Running inference step...")
    logging.info("Running inference step...")
    start = time()
    preds, self_attn_ws, cross_attn_ws = utils.model_utils.predict(
        st.session_state.model, 
        batch_input, 
        debug, 
        device
    )
    end = time()

    prediction_time = end - start
    # os.system("nvidia-smi")

    return preds, prediction_time
