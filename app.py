import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import numpy as np

model = pickle.load(open('model2.pkl' , 'rb'))

st.title("Credit Card Anomaly Detection")
st.text("This web app let's you upload transcations and dtect anomlaies in the transactions.")

uploaded_file = st.file_uploader("Upload you file here")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header("Data Header")
    st.write(df.head())
    
    object_types = df.select_dtypes(include='object').columns

    for col in object_types:
        df[col] = pd.to_numeric(df[col] , errors = 'coerce')
        df[col] = df[col].astype(float)

    df.dropna(inplace = True)
    st.header("Statistics")
    st.write(df.describe())

    scaler = MinMaxScaler()
    df[['Amount' , 'Time']] = scaler.fit_transform(df[['Amount' , 'Time']])
    # PCA
    df.loc[ :, 'V1':'V28'] = scaler.fit_transform(df.loc[ :, 'V1':'V28'])
    X_mean = df.loc[ :, 'V1':'V28'].mean()
    X_std = df.loc[ :, 'V1':'V28'].std()
    Z = (df.loc[ :, 'V1':'V28'] - X_mean) / X_std

    pca = PCA(n_components=2)
    pca.fit(Z)
    x_pca = pca.transform(Z)
    df['PC1'] = x_pca[:, 0]
    df['PC2'] = x_pca[:, 1]
    model_df = df.copy()
    model_df.drop(columns = df.loc[ :, 'V1':'V28'].columns , inplace = True)
    model_df = model_df.select_dtypes(include = 'float')
    st.header("Principal Components")
    st.write(model_df.sample())

    st.header("Predictions")
    y_pred = model.predict(model_df)
    y_pred = np.where(y_pred == -1 , 1 , 0)
    df['Class'] = y_pred
    model_df['Class'] = y_pred
    st.write(df[['Time' , 'Amount' , 'Class']].head())

    st.title('PCA Scatter Plot using Matplotlib')

    fig, ax = plt.subplots()

    scatter = ax.scatter(model_df['PC1'], model_df['PC2'], c=model_df['Class'], cmap='viridis', alpha=0.7)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Scatter Plot of Principal Components')

    legend = ax.legend(*scatter.legend_elements(), title="Categories")
    ax.add_artist(legend)

    st.pyplot(fig)

        