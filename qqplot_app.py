import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot

st.set_page_config(layout='wide')
st.title('Portugese Winery Dataset')

# loading the combined wine dataset
df = pd.read_csv('data/wine-quality.csv')

# creating a dataframe for only red wine
df_red = df[df['red or white']=='red'].copy()
df_red.reset_index(inplace=True)

# creating a dataset for only white wine
df_white = df[df['red or white']=='white'].copy()
df_white.reset_index(inplace=True)

# getting the input features
input_features = df.columns[:-4].tolist()
selected_input = st.selectbox('What input feature do you want to compare?', input_features)

col1, col2, col3= st.columns(3)

with col1:
    st.markdown("#### Both Wines")
    fig1, ax1 = plt.subplots()
    ax1 = probplot(df[selected_input],plot =plt)
    st.pyplot(fig1)

with col2:
    st.markdown("#### Red Wine")
    fig2, ax2 = plt.subplots()
    plt.title(None)
    ax2 = probplot(df_red[selected_input],plot =plt)
    st.pyplot(fig2)

with col3:
    st.markdown("#### White Wine")
    fig3, ax3 = plt.subplots()
    ax3 = probplot(df_white[selected_input],plot =plt)
    st.pyplot(fig3)
