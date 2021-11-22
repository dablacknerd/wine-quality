import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

# loading the combined wine dataset
df = pd.read_csv('data/wine-quality.csv')

# creating a dataframe for only red wine
df_red = df[df['red or white']=='red'].copy()
df_red.reset_index(inplace=True)

# creating a dataset for only white wine
df_white = df[df['red or white']=='white'].copy()
df_white.reset_index(inplace=True)

# getting the input features
input_features_1 = df.columns[:-4].tolist()

# getting input features plus target
# the target is an ordinal variable
# this will be used for Kendall's Tau correlation which can
# perform correlation with ordinal variables
input_features_2 = df.columns[:-4].tolist() + ['quality_flag']

@st.cache()
def calculate_correlation_matrix(data, method, columns):
    '''
    This function returns a correlation matrix.

    Parameter(s):
        data: A dataframe with all numerical columns
        method: The method the app will use to calculate the correlation.
                available options are pearson, spearman and kendall's tau
        column: The columns in the dataframe that will be included in the correlation matrix
    '''
    X = data[columns]
    return X.corr(method=method)

method_dict ={"Pearson":"pearson", "Spearman Rank":"spearman", "Kendall's Tau":"kendall"}
col1, col2, col3= st.columns(3)
sns.set_style('darkgrid')
custom_cmap = sns.diverging_palette(10, 255, n=3)

with col1:
    corr_method1 = st.selectbox('Correlation Method for Both', ['Pearson','Spearman Rank',"Kendall's Tau"])
    st.markdown("#### Both Wines")

    if corr_method1 == "Kendall's Tau":
        corr = calculate_correlation_matrix(df, method_dict[corr_method1], input_features_2)
    else:
        corr = calculate_correlation_matrix(df, method_dict[corr_method1], input_features_1)

    #mask = np.triu(np.ones_like(corr, dtype=bool))
    #mask=mask,
    fig1, ax1 = plt.subplots()
    ax1 = sns.heatmap(corr, cmap=custom_cmap, vmax=1.0,vmin=-1.0,annot=True,fmt='.1f')
    st.pyplot(fig1)

with col2:
    corr_method2 = st.selectbox('Correlation Method for Red Wine', ['Pearson','Spearman Rank',"Kendall's Tau"])
    st.markdown("#### Red Wine")

    if corr_method2 == "Kendall's Tau":
        corr = calculate_correlation_matrix(df_red, method_dict[corr_method2], input_features_2)
    else:
        corr = calculate_correlation_matrix(df_red, method_dict[corr_method2], input_features_1)

    #mask = np.triu(np.ones_like(corr, dtype=bool))
    #mask=mask,
    fig2, ax2 = plt.subplots()
    ax2 = sns.heatmap(corr, cmap=custom_cmap, vmax=1.0,vmin=-1.0,annot=True,fmt='.1f')
    st.pyplot(fig2)

with col3:
    corr_method3 = st.selectbox('Correlation Method for White Wine', ['Pearson','Spearman Rank',"Kendall's Tau"])
    st.markdown("#### White Wine")

    if corr_method3 == "Kendall's Tau":
        corr = calculate_correlation_matrix(df_white, method_dict[corr_method3], input_features_2)
    else:
        corr = calculate_correlation_matrix(df_white, method_dict[corr_method3], input_features_1)

    #mask = np.triu(np.ones_like(corr, dtype=bool))
    #mask=mask, 
    fig3, ax3 = plt.subplots()
    ax3 = sns.heatmap(corr,cmap=custom_cmap, vmax=1.0,vmin=-1.0,annot=True,fmt='.1f')
    st.pyplot(fig3)
