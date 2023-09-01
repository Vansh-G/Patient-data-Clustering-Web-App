import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
import streamlit as st
from finalprocessing import (
    load_local,
    load_upload,
    preprocessing,
    clustering,
    no_clusters,
)
import plotly.graph_objs as go
import plotly.express as px


# from model import (
#     no_clusters,
#     clustering)


# Loading the model
# loaded_model = pickle.load(open('Trained_model.sav', 'rb'))

# def cluster_pred(input_data):


def introduction():
    st.title("Clustering Patients based on their test Results using KMeans Clustering")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.markdown(
            """ [Source Code](https://github.com/)"""
        )
    with col2:
        st.sidebar.markdown(
            """[User Guide](https://github.com/)"""
        )
    with st.expander("""How can this app help you?"""):
        st.write(
            """This Streamlit Web Application has been designed for **Clustering**."""
        )
        st.markdown(
            """
    1. Upload a dataset or use the example  _(If you use the example you do not need to follow the next steps.)_
    2. [Parameters] select all the columns you want to keep in the analysis
    3. Click on **Start Calculation?** to launch the analysis
     
    
    """
        )


def upload_ui():
    st.sidebar.subheader("Load a Dataset ")
    st.sidebar.write("Upload your dataset (.csv)")
    # Upload
    input = st.sidebar.file_uploader("")
    if input is None:
        dataset_type = "LOCAL"
        st.sidebar.write(
            "_If you do not upload a dataset, an example is automatically loaded to show you the features of this app._"
        )
        df_knn = load_local()
        list_var = dataset_ui(df_knn, dataset_type)
    else:
        dataset_type = "UPLOADED"
        with st.spinner("Loading data.."):
            df_knn = load_upload(input)
            st.write(df_knn.head())

            # df_knn = pd.DataFrame()
        list_var = dataset_ui(df_knn, dataset_type)

    # Process filtering
    st.write("\n")
    st.subheader(""" Your dataset with the final version of the features""")
    df_knn = df_knn[list_var].copy()
    st.write(df_knn.head(2))

    return list_var, dataset_type, df_knn


def dataset_ui(df_knn, dataset_type):
    # SHOW PARAMETERS
    expander_default = dataset_type == "UPLOADED"

    st.subheader(" Please choose the following features in your dataset")
    with st.expander("FEATURES TO USE FOR THE ANALYSIS"):
        st.markdown(
            """
        _Select any **8** columns that you want to include in the analysis(including id)._
    """
        )
        dict_var = {}
        dict_var[df_knn.columns[0]] = st.checkbox("{} (IN/OUT)".format(df_knn.columns[0]), value=1)
        for column in df_knn.columns[1:]:
            dict_var[column] = st.checkbox("{} (IN/OUT)".format(column), value=0)
    filtered = filter(lambda col: dict_var[col] == 1, df_knn.columns)
    list_var = list(filtered)

    return list_var


# Set page configuration
st.set_page_config(
    page_title="Statistical Clustering of Patients",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Page Title
introduction()


# Set up the page
@st.cache(
    persist=False,
    allow_output_mutation=True,
    suppress_st_warning=True,
    show_spinner=True,
)
# Preparation of data
def prep_data(df):
    col = df.columns
    return col


# -- Page
st.cache_data.clear()

# Information about the Dataset
st.header("**Information about the Dataset**")

# Upload Data Set
list_var, dataset_type, df_knn = upload_ui()

# Start Calculation ?
start_calculation = st.checkbox("Start Calculation?", key="show", value=False)
    
    # start_calculation = True
#   else:
#     if dataset_type == "LOCAL":
#         start_calculation = True
#     else:
#         start_calculation = False

# # Process df for uploaded dataset
# if dataset_type == "UPLOADED" and start_calculation:
#     df_knn = preprocessing(df_knn)


if start_calculation:
    st.header("**Clustering**")
    df_knn.set_index('id')

    df_knn = preprocessing(df_knn)
    
    no_clusters(df_knn)
    n = int(st.text_input('Enter the no. of clusters',value = 1))
    if n:
        st.write("You entered: ", n)
    pred = clustering(n,df_knn)
    st.write(pred)

    df_knn['Clusters'] = pred

    cluster_names = [f"Cluster {k}" for k in np.unique(df_knn["Clusters"])]
    print(df_knn.groupby('Clusters').mean())
    print(list_var)
    data = [
        go.Bar(name=f, x=cluster_names, y=df_knn.groupby("Clusters")[f].mean()) for f in list_var ]  # a list of plotly GO objects with different Y values
    fig = go.Figure(data=data)
            # Change the bar mode
    fig.update_layout(barmode="group", height =700, width = 1300)
    # left, middle, right = st.columns((0, 5, 2))
    # with middle:
    st.plotly_chart(fig)
