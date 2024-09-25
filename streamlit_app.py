import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title('😁DSAIC-Deploying-AI-model')

st.info('This app builds a AI model on penguine dataset')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('penguins_data.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw
