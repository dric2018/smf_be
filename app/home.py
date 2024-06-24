import app_utils

import os
import sys
sys.path.append("../src") # import all RTCRAM source code

import config

import pandas as pd

from PIL import Image

import streamlit as st
from time import time

import utils

st.set_page_config(
    page_title="S-JEP Demo App",
    page_icon="🧠",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

app_utils.init_session_states()

st.write("## Welcome to the S-JEP 🦾 Working Associative Memory 🧠 Demo App!")

st.sidebar.warning("Select a page above.")

model_img = Image.open("../imgs/SMF-BE.jpg")
st.image(model_img, caption="An example generation of a Behavioral Episode (BE)")

@st.cache_data
def load_data():
    df = pd.read_csv(st.session_state.DATA_PATH)[st.session_state.COLUMNS]
    return df

with st.spinner("Loading data frame..."):
    data = load_data()

# save data file to session state
st.session_state.data = data