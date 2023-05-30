import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats

# from scipy.stats import shapiro
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
import plotly.express as px
import plotly.graph_objs as go


# Load data from user upload
def load_upload(input):
    # Create Data Frame
    df_knn = pd.read_csv(input)

    return df_knn


def load_local():
    # Create Data Frame
    df_knn = pd.read_csv("Final Sheet.csv")

    # Sample data used to show the head
    # global df
    # df = pd.read_csv('sample.csv',sep=None ,engine='python', encoding='utf-8',
    #             parse_dates=True,
    #             infer_datetime_format=True, index_col=0)

    return df_knn


def preprocessing(df_knn):

    encoded_df = df_knn.copy()
    st.write(encoded_df.info())
    for col in encoded_df.columns:
        
        if encoded_df[col].dtype == "object" or pd.api.types.is_categorical_dtype(
            encoded_df[col]
        ):
            encoded_df[col] = encoded_df[col].astype("category").cat.codes
            encoded_df[col] = encoded_df[col].astype(int)


    rob_scaler = RobustScaler()
    data = rob_scaler.fit_transform(encoded_df)

    dataset = pd.DataFrame(data)
    dataset.columns = encoded_df.columns
    
    dataset.dropna(axis=0,inplace = True)

    return dataset


def no_clusters(df):
    ssd = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i)
        clusters = km.fit_predict(df)
        ssd.append(km.inertia_)

    # df["Cluster"] = clusters
    # fig, ax = plt.subplots()
    # ax.plot(range(1, 11), ssd)
    # fig = px.line(x=range(1,11),y=ssd)
    # fig = go.Figure(data = go.Scatter(x=range(1,11),y=ssd))
    # fig,ax = plt.subplots()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(1,11)], y=ssd)) 
    # fig.show()
    left, middle, right = st.columns((2, 5, 2))
    with middle:
        st.plotly_chart(fig)




def clustering(n, X):
    km = KMeans(n_clusters=n)
    km.fit(X)
    clusters1 = km.predict(X)

    return clusters1


# filename = 'Trained_model.sav'
# pickle.dump(km,open(filename,'wb'))
