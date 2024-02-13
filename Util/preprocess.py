import statsmodels.api as sm
import streamlit as st
import pandas as pd
from Util import outlier as out 
from Util import missing_value as mv
from Util import encoding as ed
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

def encode_categorical_columns(df, col4):
    col4.subheader("Feature Encoding")
    encoded_df= df.copy()
    #st.dataframe(df)
    categorical_columns = df.select_dtypes(include=['object']).columns 
    # Get columns and their data types
    
    column_data_types= df[categorical_columns].dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column Name'})
    
    
    
    # Create a new column for Imputation Method in DataFrame
    column_data_types['Encoding Method'] = [''] * len(column_data_types)
    # Display dropdowns for selecting imputation method for each column   
   
    for i, row in column_data_types.iterrows():
        if row['Column Name'] != 'ADDRESS':
            col = row['Column Name']
            data_type = row['Data Type']
            # Customize dropdown options based on data type
            
            encoding_methods = ['Select', 'One-Hot Encoding', 'Label Encoding', 'Ordinal Encoding', 'Frequency Encoding']  # Add more methods as needed for object type
            
            encoding_method = col4.selectbox(f'Select Encoding Method for {col} ({data_type}):', encoding_methods,
                                            key=col + str(i+1))

            # Update the DataFrame with the selected imputation method
            column_data_types.loc[column_data_types['Column Name'] == col, 'Encoding Method'] = encoding_method 
    if (column_data_types['Encoding Method'] != 'Select').any():
        for _, row in column_data_types.iterrows():            
            col = row['Column Name']            
            encoding_method = row['Encoding Method']
            
            if encoding_method == 'One-Hot Encoding':
                encoded_df = pd.get_dummies(encoded_df, columns=[col], prefix=[col],dtype= int)
            elif encoding_method == 'Label Encoding':
                encoded_df[col]=encoded_df[col].astype('category').cat.codes
            elif encoding_method == 'Ordinal Encoding':
                pass
            elif encoding_method == 'Frequency Encoding':
                freq_map = df[col].value_counts(normalize=True).to_dict()
                encoded_df[col] = df[col].map(freq_map)       
    else:
        st.error('Error: Please select imputation for all columns.') 
    scaling(encoded_df,col4)
    #st.dataframe(encoded_df)
    #scaling(encoded_df,col4)

def missing_value(df,col2,col3,col4,col_list,target_column):
    col2.subheader("Missing Value Treatment")
    df_mv = df[col_list]   
    # Get columns and their data types
    column_data_types = mv.get_column_data_types(df_mv)
    # Create a new column for Imputation Method in DataFrame
    column_data_types['Imputation Method'] = [''] * len(column_data_types)
    # Display dropdowns for selecting imputation method for each column   
    for _, row in column_data_types.iterrows():
        if row['Column Name'] != 'ADDRESS':
            col = row['Column Name']
            data_type = row['Data Type']
            # Customize dropdown options based on data type
            if data_type == 'object':
                imputation_methods = ['Select', 'auto', 'most_frequent', 'classification']  # Add more methods as needed for object type
            else:
                imputation_methods = ['Select', 'auto', 'mean', 'most_frequent', 'median', 'max', 'min', 'regression', 'KNN']  # Add more methods as needed for numeric type

            imputation_method = col2.selectbox(f'Select Imputation Method for {col} ({data_type}):', imputation_methods,
                                            key=col)

            # Update the DataFrame with the selected imputation method
            column_data_types.loc[column_data_types['Column Name'] == col, 'Imputation Method'] = imputation_method           
    if (column_data_types['Imputation Method'] != 'Select').any():
        for _, row in column_data_types.iterrows():            
            col = row['Column Name']            
            imputation_method = row['Imputation Method']
            if imputation_method != 'Select':
                if imputation_method == 'auto':
                    if row['Data Type'] == 'object':
                        df_mv= mv.simple_imputation(df_mv,col, 'most_frequent')
                    else:
                        df_mv=mv.simple_imputation(df_mv,col, 'mean')
                if imputation_method == 'mean' or imputation_method == 'median' or imputation_method == 'most_frequent':
                    df_mv=mv.simple_imputation(df_mv,col, imputation_method)
                elif imputation_method == 'max':
                    df_mv= mv.max_imputation(df_mv,col)
                elif imputation_method == 'min':
                    df_mv= mv.min_imputation(df_mv,col)
                elif imputation_method == 'regression':
                    
                    df_mv= mv.linear_regression_imputation(df_mv, col)                    
                elif imputation_method == 'classification':
                    df_mv= mv.logistic_regression_imputation(df_mv, col)
                elif imputation_method == 'KNN':
                        
                    df_mv= mv.KNN_imputation(df_mv, col)              
    else:
        st.error('Error: Please select imputation for all columns.')    
    dict1 = outlier_treatment(df_mv,col3,col4,target_column)    
    return dict1

def outlier_treatment(df_mv,col3,col4,target_column):
    col3.subheader("Outlier Treatment")
    df= df_mv.copy()
    # st.dataframe(df)
    # st.title('Treating Outliers Configuration')

    # Get columns and their data types
    columns_with_outliers = out.detect_outliers(df)

    # Create a new column for Imputation Method in DataFrame
    # Convert the list to a DataFrame
    outliers_df = pd.DataFrame(columns_with_outliers, columns=['Column Name'])
    outliers_df['Treatment Method'] = [''] * len(outliers_df)

    # Display table with columns, data types, and dropdowns
    # st.write('### DataFrame with Treatment Configuration:')

    # Display dropdowns for selecting imputation method for each column
    
    for i,row in outliers_df.iterrows():
        
        col = row['Column Name']
        
        
        treatment_methods = ['Select', 'auto', 'IQR', 'zscore','svm','knn','isolation_forest']  # Add more methods as needed for object type
        
        treatment_method = col3.selectbox(f'Select Treatment Method for {col} :', treatment_methods,
                                        key=col + str(i))

        # Update the DataFrame with the selected imputation method
        outliers_df.loc[outliers_df['Column Name'] == col, 'Treatment Method'] = treatment_method

    # # Display the updated DataFrame
    # st.write('### Updated DataFrame with Treatment Configuration:')
    # st.table(outliers_df)

    # Button to trigger imputation
    if (outliers_df['Treatment Method'] != 'Select').any():
        for _, row in outliers_df.iterrows():            
            col = row['Column Name']            
            treatment_method = row['Treatment Method']
            if treatment_method != 'Select':
                if treatment_method == 'auto': 
                    df= out.auto_treatment(df,col)
                elif treatment_method == 'IQR':                        
                    df= out.treat_and_visualize_outliers(df,col,'IQR')                            
                elif treatment_method == 'zscore': 
                    df= out.treat_and_visualize_outliers(df,col,'z-score')     
                elif treatment_method == 'isolation_forest': 
                    df= out.treat_and_visualize_outliers(df,col,'isolation_forest')                     
                elif treatment_method == 'svm': 
                    df= out.treat_and_visualize_outliers(df,col,'svm') 
                
                elif treatment_method == 'knn': 
                    df= out.treat_and_visualize_outliers(df,col,'knn')
        dict1 = dict1 = encode_categorical_columns(df,col4)
        return dict1
        
        
    else:
        st.error('Error: Please select treatment for all columns.')
       
    
def col_include(df_original,col1,col2,col3,col4,target_column):
    col1.subheader("Feature Selection")
    df= df_original.copy()
    #
    df_data = pd.DataFrame({
        'Column_Name': df.columns,
        'Data_Type': df.dtypes.values
    })
    df_with_selections = df_data.copy()
    df_with_selections.insert(0, "Include", False)
    df_with_selections['Include'] = df_with_selections['Column_Name'] == target_column
    
    # Get dataframe row-selections from user with st.data_editor
    
    edited_df = col1.data_editor(
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
        dict1= missing_value(df,col2,col3,col4,col_list,target_column)
        #dict1 = encoding(df,col_list)  
        #return col_list
        return dict1

def encoding(df,target_column):
    
    scaler = MinMaxScaler()
    #numerical_columns = df[col_list].select_dtypes(exclude =['object']).columns 
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Fit the scaler to the data and transform the data
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True,dtype=int)
    scaled_data = scaler.fit_transform(df_encoded)
    # Convert the scaled data back to a DataFrame (optional)
    scaled_df = pd.DataFrame(scaled_data, columns=df_encoded.columns)
    #   st.dataframe(scaled_df)
    # Perform one-hot encoding if categorical columns are presen
    dict1 = backward_elimination(scaled_df,target_column)
    # Convert boolean values to integers (1 or 0)
    return dict1

def scaling(df,col4):
    df_encoded = df.copy()
    col4.subheader("Feature scaling")
    scaler = col4.selectbox("Select scaling Method:",['Select','Min-max Scaling','Standarization','Robust Scaling'])
    
    if scaler == 'Standarization':
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Fit the scaler to your data and transform it
        scaled_data = scaler.fit_transform(df_encoded)
        
        
    
   
    if scaler == 'Min-max Scaling':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_encoded)
        
        
    if scaler == 'Robust Scaling':
        # Create a RobustScaler object
        scaler = RobustScaler()

        # Fit the scaler to your data and transform it
        scaled_data = scaler.fit_transform(df_encoded)
    if scaler!='Select':   
        # Convert the scaled data back to a DataFrame (optional)
        scaled_df = pd.DataFrame(scaled_data, columns=df_encoded.columns) 
    
        scaled_df.to_csv('/Users/nishagarg7/Downloads/3i- infotech/AutoML_Services-main/Experiments/Transformed.csv')
        
        #st.dataframe(scaled_df)
def backward_elimination(df_model,target_column):
    df = df_model.copy()
    # Assume 'X' is your feature matrix and 'y' is your target variable
    # Replace them with your actual feature matrix and target variable
    st.write(target_column)
    X = df.drop([target_column,'Unnamed: 0'], axis=1)
    
    y = df[target_column]

    # Add a constant column to the feature matrix (required for statsmodels)
    X = sm.add_constant(X)
    
    # Fit the initial model
    model = sm.OLS(y, X).fit()

    # Perform backward elimination
    while True:
        # Get the p-values for all features
        p_values = model.pvalues[1:]  # Exclude the constant term
        #st.write(p_values)
        # Find the feature with the highest p-value
        max_p_value = p_values.max()
        if max_p_value > 0.05:  # Set your significance level
            
            # Remove the feature with the highest p-value
            feature_to_remove = p_values.idxmax()
            X = X.drop(feature_to_remove, axis=1)
            
            # Fit the updated model
            model = sm.OLS(y, X).fit()
        else:
            break  # Exit the loop if no feature has a p-value greater than 0.05
    
    my_dict = dict(zip(X.columns.tolist()[1:], model.pvalues[1:]))
    
    return my_dict











