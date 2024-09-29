import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import io

DATA_URL = 'data_kaggle.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Streamlit\.venv\Scripts\data_kaggle.csv')
    return data

data = load_data()

st.title('Property Listings in Kuala Lumpur')

st.header('--==Data Understanding==--')
st.subheader('Raw Dataset')
st.write(data.head())

st.subheader('Raw Dataset Information')
st.write('Description')
st.write(data.describe(include='all'))
st.write('------')
st.write('Number of uniques')
st.write('Unique Location :', data['Location'].nunique())
st.write('Unique Price :', data['Price'].nunique())
st.write('Unique Rooms :', data['Rooms'].nunique())
st.write('Unique Property Type :', data['Property Type'].nunique())
st.write('Unique Size :', data['Size'].nunique())
st.write('Unique Furnishing :', data['Furnishing'].nunique())
st.write('------')
st.write('Number of null values')
st.write(data.isna().sum())
st.write('------')
st.write('Number of duplicates')
st.write(data.duplicated().sum())

st.header('--==Data Preparation==--')
st.subheader('Sampling')
df_sample = data.sample(frac=0.2)
df_sample = df_sample.reset_index()
df_sample = df_sample.drop(['index'], axis=1)
st.write(df_sample.describe(include='all'))

st.subheader('Mengubah Price Menjadi Numeric')
df_price = df_sample.copy()
df_price = df_price['Price'].replace({'RM ': '', ',': ''}, regex=True).astype(float)
df_floatPrice = df_sample.copy()
df_floatPrice.Price = df_price
st.write(df_floatPrice.head())