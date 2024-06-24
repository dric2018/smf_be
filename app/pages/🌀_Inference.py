
from app_utils import load_sample, test_model_io

import numpy as np

import os
import sys
sys.path.append("../src") # import all RTCRAM source code

import config

from PIL import Image

import streamlit as st

from time import time

import utils

st.write("## BE Model Inference 🌀!")

load_sample_btn = st.button("Load Sample")

if load_sample_btn:
    in_state, action_desctiption = load_sample()
    in_state = np.array(in_state)

    if in_state is not None:
        st.session_state.in_state = in_state
        st.session_state.action_desctiption = action_desctiption

if st.session_state.in_state is not None:
    st.image(st.session_state.in_state, caption="Scene view", width=config.IMG_SIZE)

ad = st.text_input(label="Action description", value=st.session_state.action_desctiption)
if ad:
    ad = ad.lower()
run_btn = st.button("Run Test")

if run_btn:
    if ad not in [None, " ", ""]:
        st.success(f"Running regular inference pipeline on {config.DEVICE} device...")
        with st.spinner("Preparing model checkpoint..."):
            t0                              = time()
            while st.session_state.model is None:
                
                st.session_state.model      = utils.model_utils.load_checkpoint(model_name="RTCRAM_final")
                t1                          = time()
                st.session_state.model_loading_time = t1 - t0

        st.warning(f"Loaded model in {st.session_state.model_loading_time:.3f}s")            

        with st.spinner("Running Test"):
            preds, prediction_time = test_model_io(instruction=ad, img=st.session_state.in_state)
            pred_0 = "".join(preds[0])
            st.info(f"Instruction: {ad}")
            st.success(f"Predicted motor commands: `{pred_0}`")
            st.warning(f"Prediction time: {prediction_time:.3f}s")            
    else:
        st.success("Running alternative inference pipeline...")
        with st.spinner("Running Test"):
            preds, prediction_time = test_model_io(img=st.session_state.in_state)
            pred_0 = "".join(preds[0])
            st.success(f"Predicted motor commands: `{pred_0}`")
            st.warning(f"Prediction time: {prediction_time:.3f}s")   