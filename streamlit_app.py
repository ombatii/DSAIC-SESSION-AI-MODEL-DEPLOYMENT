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
  df = pd.read_csv('https://raw.githubusercontent.com/ombatii/DSAIC-Deploying-AI-model/refs/heads/master/penguins_data.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw


with st.expander('Data visualization'):
    # Scatter chart: Bill length vs Body mass, colored by species
    st.subheader("Scatter Plot: Bill Length vs Body Mass")
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

    # Bar chart: Mean body mass by species
    st.subheader("Bar Chart: Mean Body Mass by Species")
    mean_body_mass = df.groupby('species')['body_mass_g'].mean().reset_index()
    st.bar_chart(data=mean_body_mass, x='species', y='body_mass_g')

    # Histogram: Distribution of flipper lengths
    st.subheader("Histogram: Distribution of Flipper Length")
    fig, ax = plt.subplots()
    sns.histplot(df['flipper_length_mm'], kde=True, ax=ax)
    ax.set_title('Distribution of Flipper Length')
    st.pyplot(fig)



# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
