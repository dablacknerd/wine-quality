import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests

st.set_page_config(layout='wide')



df = pd.read_csv('data/wine-quality.csv')
input_features = df.columns[:-4].tolist()
X = df[input_features]
corr = X.corr()

df_red = df[df['red or white']=='red'].copy()
df_red.reset_index(inplace=True)

df_white = df[df['red or white']=='white'].copy()
df_white.reset_index(inplace=True)



st.title('Wine Quality Dataset')

input_features = df.columns[:-2].tolist()
select_inputs = st.sidebar.selectbox('What input feature do you want to examine?', input_features)

with st.container():
    col1, col2= st.columns([3,3])
    sns.set_style('darkgrid')

    with col1:
        fig1, ax1 = plt.subplots()
        custom_cmap = sns.diverging_palette(10, 255, n=3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.title("Diagonal Correlation Matrix")
        ax1 = sns.heatmap(corr,mask=mask,cmap=custom_cmap, vmax=1.0,vmin=-1.0,annot=True,fmt='.1f')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        wine_type_color = {'white':'blue','red':'red'}
        plt.title("Outlier Visualization for {}".format(select_inputs.title()))
        ax2 = sns.boxplot(x='red or white', y=select_inputs, data=df, palette=wine_type_color)
        plt.xlabel(None)
        plt.ylabel(None)
        st.pyplot(fig2)
