import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_component
from PIL import Image
from Util import exploratory_data_analysis as eda
from Util import dataset_utils
from Util import missing_value as mv
from Util import Modelling as md
import os

from Util import preprocess as pp

pageicon = Image.open(r"/Users/nishagarg7/Downloads/3i- infotech/AutoML_Services-main/Images/3i_icon.png") 
st.set_page_config(page_title="FutureTech AutoML",
                   page_icon=pageicon,
                   layout='wide')
# Page Title
title_col1, title_col2, title_col3 = st.columns(3)
with title_col2:
    st.title("*FutureTech AutoML*")

st.title("")
st.title("")
st.title("")

if 'create_dataset' not in st.session_state:
    st.session_state.create_dataset = False
def click_create_dataset():
    st.session_state.create_dataset = True

if 'analyze_dataset' not in st.session_state:
    st.session_state.analyze_dataset = False
def click_analyze_dataset():
    st.session_state.create_dataset = False
    st.session_state.analyze_dataset = True
    
if 'training' not in st.session_state:
    st.session_state.training = False
def click_training():
    st.session_state.training = True

if 'prediction' not in st.session_state:
    st.session_state.prediction = False
def click_prediction():
    st.session_state.prediction = True

if 'auto_eda' not in st.session_state:
    st.session_state.auto_eda = False
def click_auto_eda():
    st.session_state.auto_eda = True

if 'advanced_auto_eda' not in st.session_state:
    st.session_state.advanced_auto_eda = False
def click_advanced_auto_eda():
    st.session_state.advanced_auto_eda = True

if 'create_eda' not in st.session_state:
    st.session_state.create_eda = False
def click_create_eda():
    st.session_state.auto_eda = False
    st.session_state.advanced_auto_eda = False
    st.session_state.create_eda = True

if 'create_charts' not in st.session_state:
    st.session_state.create_charts = False
def click_create_charts():
    st.session_state.create_charts = True

def main():            
    button_col1, button_col2, button_col3 = st.columns(3)
    # Dataset Upload Option
    with button_col1:
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.subheader("Prepare Your Dataset")
        st.write("Collect and prepare your data to train the model")
        col1, col2 = st.columns([0.3, 0.7])
        with col2:
            create_dataset_button = st.button("Create Dataset", on_click=click_create_dataset)

    # #Exploratory Data Analysis Button
    # with button_col2:
    #     col1, col2 = st.columns([0.2, 0.8])
    #     with col2:
    #         st.subheader("Analyze Your Data")
    #     st.write("Do exploratory data analysis on your data to get hidden and meaningful inferances from it.")
    #     col1, col2 = st.columns([0.25, 0.75])
    #     with col2:
    #         analyze_button = st.button("Anlyze Dataset", on_click=click_analyze_dataset)
            
    # Create Model Option
    with button_col2:
        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            st.subheader("Train Your Model")
        st.write("Train the best-in-class machine learning model with your dataset")
        col1, col2 = st.columns([0.25, 0.75])
        with col2:
            train_model_button = st.button("Start Model Training", on_click=click_training)
        
    # Prediction Option
    with button_col3:
        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            st.subheader("Get Predictions")
        st.write("After you train the model, you can use it to get predictions, either online or batch.")
        col1, col2 = st.columns([0.35, 0.65])
        with col2:
            prediction_button = st.button("Predict", on_click=click_prediction)
    
    
    # Dataset Upload        
    if st.session_state.create_dataset:
        "---"  
        dataset_utils.make_dataset()
        if not dataset_utils.upload_checker():
            global df 
            df = dataset_utils.fetch_dataframe()
        st.title("")
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns([0.2, 0.1, 0.15, 0.15, 0.12, 0.12, 0.15])
        # Data Preview Button
        with col2:    
            preview_button = st.button("Preview")
        # Auto EDA Button
        with col3:
            auto_eda_button = st.button("Get EDA Report", on_click=click_auto_eda)
        # Manual EDA Button
        with col4:
            manual_eda_button = st.button("Create EDA Report", on_click=click_create_eda)
        # Quick Training Button   
        with col5:
            quick_training_button = st.button("Quick Train", key="quick_train1")
        # Training Button   
        with col6:
            training_button = st.button("Train Model", key="train1")
        
        # Dataset Preview 
        if preview_button:
            st.title('')
            dataset_utils.get_dataset_preview()
                
        # Automated Exploratory Data Analysis    
        elif st.session_state.auto_eda == True:
            if dataset_utils.upload_checker():
                error = dataset_utils.upload_checker()
                st.write(f":red[{error}]")          
            else:
                st.title('')
                eda.automated_eda_report(df)
                
                # Advanced EDA Button
                advanced_eda_button = st.button("Show Advanced EDA", on_click=click_advanced_auto_eda)
                if st.session_state.advanced_auto_eda == True:
                    eda.advanced_eda(df)
                    
                st.title("")
                
                col1, col2, col3, col4 = st.columns([0.37, 0.15, 0.15, 0.33])
                with col2:
                    quick_training_button = st.button("Quick Train", key="quick_train2")
                # Training Button   
                with col3:
                    training_button = st.button("Train Model", key="train2")
                
        # Manual Exploratory Data Analysis    
        elif st.session_state.create_eda == True:
            st.session_state.auto_eda = False
            if dataset_utils.upload_checker():
                error = dataset_utils.upload_checker()
                st.write(f":red[{error}]")          
            else:
                "---"
                col1, col2, col3 = st.columns([0.33, 0.4, 0.3])
                with col2:
                    st.header("**Exploratory Data Analysis**")
                st.subheader("")
                
                col1, col2, col3, col4, col5, col6, col7 = st.columns([0.15, 0.12, 0.18, 0.12, 0.12, 0.15, 0.15])
                with col2:
                    summary_eda_button = st.button("Summary")
                with col3:
                    missing_val_eda_button = st.button("Missing Value Analysis")
                with col4:
                    corr_matrics_button = st.button("Correlation")
                with col5:
                    stats_button = st.button("Statistics")
                with col6:
                    st.button("Create Charts", on_click=click_create_charts)
                
                # Dataset Summary
                if summary_eda_button:
                    st.title('')
                    col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
                    with col2:
                        st.subheader("Dataset Summary")
                    eda.get_dataset_summary(df)
                
                # Missing Value analysis
                elif missing_val_eda_button:
                    st.title('')
                    eda.get_missing_val_analysis(df)
                    
                # Corelation Matrics
                elif corr_matrics_button:
                    st.title('')
                    eda.get_correlation_matrics(df)
                
                # Discriptive Statistics
                elif stats_button:
                    st.title('')
                    col1, col2, col3 = st.columns([0.38, 0.4, 0.2])
                    with col2:
                        st.subheader("Discriptive Statistics")
                    eda.get_discriptive_stats(df)
                    
                # Charts & Graphs
                elif st.session_state.create_charts:
                    st.title('')
                    col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
                    with col2:
                        x_col = st.selectbox("X-Axis", options=df.columns)
                    with col3:
                        y_col = st.selectbox("Y-Axis", options=df.columns)
                    with col4:
                        chart = st.selectbox("Chart", options=['Line', 'Bar', 'Scatter', 'Pie', 'Heatmap', 'Histplot', 'Countplot'])
                                        
                    col1, col2, col3, col4 = st.columns([0.3, 0.2, 0.2, 0.3])
                    with col2:
                        if chart == 'Heatmap':
                            val_col = st.selectbox("Values", options=df.columns)
                        elif chart == 'Pie':
                            pass
                        else:
                            grp_options = ['']
                            grp_options.extend(df.columns.to_list())
                            grp_col = st.selectbox("Group By", options=grp_options)
                    with col3:
                        if chart == 'Countplot':
                            aggregator = st.selectbox("Aggregation", options=['count', 'percent', 'proportion', 'probability'])
                        elif chart == 'Histplot':
                            aggregator = st.selectbox("Aggregation", options=['density', 'count', 'percent', 'proportion', 'probability', 'frequency'])
                        elif chart == 'Scatter':
                            size_options = ['']
                            size_options.extend(df.columns.to_list())
                            size_col = st.selectbox("Size By", options=size_options)
                        elif chart == 'Pie':
                            pass
                        else:              
                            aggregator = st.selectbox("Aggregation", options=['', 'mean', 'median', 'sum', 'count', 'max', 'min'])
                                                                
                    col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                    with col2:
                        show_button = st.button("Show Chart")
                        
                    if show_button:
                        if chart in ['Histplot', 'Countplot', 'Pie']:
                            if x_col != y_col:
                                st.write(":green[For this type of chart only one column is needed. Keep both X-axis and Y-axis Same]")
                            else:
                                if chart == 'Pie':
                                    eda.get_pieplot(df, x_col)
                                elif chart == 'Histplot':
                                    eda.get_histplot(df, x_col, grp_col, aggregator)
                                else:
                                    eda.get_countplot(df, x_col, grp_col, aggregator)              
                        else:
                            if chart == 'Line':
                                eda.get_lineplot(df, x_col, y_col, grp_col, aggregator)
                            elif chart == 'Scatter':
                                eda.get_scatterplot(df, x_col, y_col, grp_col, size_col)
                            elif chart == 'Bar':
                                eda.get_barplot(df, x_col, y_col, grp_col, aggregator)
                            elif chart == 'Heatmap':
                                eda.get_heatmap(df, x_col, y_col, val_col, aggregator)   

    if st.session_state.training:        
        "---"
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        with col2:
            st.header("Model Training")
                
        col2, col2, col3, col4, col5, col6 = st.columns([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        with col2:
            exp_name = st.text_input("Experiment Name")
        with col3:
            dataset_list = ['']
            dataset_list.extend([file.split('.')[0] for file in os.listdir("Datasets") if file.endswith('.csv')])
            dataset_name = st.selectbox('Dataset', options=dataset_list)
            if dataset_name != '':
                df = pd.read_csv(f"Datasets/{dataset_name}.csv")
                target_option = df.columns.tolist()
            else:
                target_option = ['']
        with col4:
            target = st.selectbox('Target', options=target_option)
        with col5:
            ml_type = st.selectbox('ML Problem Type', options=['Regression', 'Classification', 'Forecasting', 'Unsupervised'])
        
        st.title("")
        
        if bool(exp_name) & bool(dataset_name) & bool(target) & bool(ml_type):
            col1,col2,col3,col4= st.columns(4)
            # col1.subheader("Missing Value Treatment")
            # col2.subheader("Outlier Treatment")
            # col3.subheader("Feature Selection")
            # target_column= 'TARGET(PRICE_IN_LACS)'
            pp.col_include(df,col1,col2,col3,col4,target)
       
            exp_list = ['']
            exp_list.extend([file.split('.')[0] for file in os.listdir("Experiments") if file.endswith('.csv')])
            col1,col2,col3= st.columns(3)
            exp_name = col1.selectbox('Transformed Dataset', options=exp_list)
            select_method =col2.selectbox('Feature Selection',['Backward Elimination','Manual Selection'])
            if exp_name != '':
                df_t = pd.read_csv(f"Experiments/{exp_name}.csv")
                
            if select_method=='Backward Elimination':
                dict1= pp.backward_elimination(df_t,target)
                st.write(dict1)
                col_list= list(dict1.keys())
                
            elif select_method=='Manual Selection':
                df_data = pd.DataFrame({
                'Column_Name': df_t.columns,
                'Data_Type': df_t.dtypes.values
            })
                df_with_selections = df_data.copy()
                df_with_selections.insert(0, "Include", False)
                df_with_selections['Include'] = df_with_selections['Column_Name'] == target
                
                # Get dataframe row-selections from user with st.data_editor
                
                edited_df = col2.data_editor(
                    df_with_selections, 
                    hide_index=True,
                    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                    disabled=df.columns,
                    num_rows="dynamic",
                )  
                 # Filter the dataframe using the temporary column, then drop the column
                selected_rows = edited_df[edited_df.Include] 
                if (edited_df['Include']== 1).any():
                    selected_rows.drop('Include', axis=1)
                    col_list = selected_rows['Column_Name'].tolist() 
            st.write(col_list)
            st.dataframe(df_t)   
            
            
            
                    
    # elif train_model_button | prediction_button:
    #     st.write(":red[Developement under Progress... Please try after some time.]")
              
if __name__ == '__main__':
    main()
            
        