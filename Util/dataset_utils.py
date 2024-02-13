import pandas as pd
import numpy as np
import streamlit as st 

###################################################################################################################
# Create Dataset from uploaded file
def make_dataset():
    title_col1, title_col2= st.columns([0.42, 0.58])
    with title_col2:
        st.subheader("Create Dataset")
        
    col1, col2 = st.columns(2)
    # Datset Name
    with col1:
        global dataset_name
        global data_type
        global dataset_desc
        
        dataset_name = st.text_input("Dataset Name", max_chars=100)
        data_type = st.selectbox("Select Data Type", options=['Tabular', 'Text', 'Image', 'Video'], help="Select the type of data your datset will contain")
    
    # Dataset Descriptiom
    with col2:
        dataset_desc = st.text_area("Description", height=120, max_chars=500)
    # Data Scource
    data_source = st.radio("Select Data Source", options=["Upload file from your computer",
                                                          "Upload file from cloud storage",
                                                          "Select table or view from database"])
    if data_source == "Upload file from your computer":
        global data_file
        data_file = st.file_uploader("Upload CSV/Excel file from your computer")
        
        global df
        if data_file:
            try:
                df = pd.read_csv(data_file)
            except:
                df = pd.read_excel(data_file)
            
            # Storing the Data as csv   
            df.to_csv(f"/Users/nishagarg7/Downloads/3i- infotech/AutoML_Services-main/Datasets/{dataset_name}.csv", index=False)
#######################################################################################################

######################################################################################################
def fetch_dataframe():
    df = pd.read_csv(f"Datasets\{dataset_name}.csv")
    return df
######################################################################################################

#######################################################################################################
# Function to Check uploaded data
def upload_checker():
    if not dataset_name:
        return "Pass the Dataset Name"
    elif not data_type:
        return "Provide the Type of Data in Dataset"
    elif not data_file:
        return "Upload file to create a dataset"
    else:
        return False
######################################################################################################

#######################################################################################################
# Dataset Preview
def get_dataset_preview():
    if upload_checker():
        error = upload_checker()
        st.write(f":red[{error}]")          
    else:               
        # Dataset Summary
        total_cols = len(df.columns)
        total_rows = df.shape[0]
        num_cols = len(df._get_numeric_data().columns)   
        catg_cols = total_cols - num_cols
        
        st.markdown("**Dataset Summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"Total Columns :    {total_cols}")
            st.markdown(f"Total Rows :       {total_rows}")
        with col2:
            st.markdown(f"Numeric Columns :    {num_cols}({np.round(num_cols/total_cols * 100,2)}%)")
            st.markdown(f"Categorical Columns :       {catg_cols}({np.round(catg_cols/total_cols * 100,2)}%)")
        
        st.dataframe(df.head(50))
###############################################################################################

