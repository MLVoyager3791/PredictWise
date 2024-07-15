import pandas as pd
import streamlit as st
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split

# Upload file
uploaded_file = st.file_uploader("Upload a CSV, TSV, or Excel file", type=["csv", "tsv", "xls", "xlsx"])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    
     # Data cleaning
    st.subheader("Data Cleaning")
    null_counts = df.isnull().sum()
    st.write("Null counts:")
    st.write(null_counts)
    
    # Handling missing values
    if df.isnull().sum().sum() > 0:
        st.write("Choose how to handle missing values:")
        replace_option = st.radio("Replace with:", ('Mean', 'Median', 'Drop'))
        
        if replace_option == 'Mean':
            df = df.fillna(df.mean())
        elif replace_option == 'Median':
            df = df.fillna(df.median())
        elif replace_option == 'Drop':
            df = df.dropna()
    
    # Select X and Y variables
    st.subheader("Select X and Y Variables")
    X_columns = st.multiselect("Select X variables:", df.columns)
    Y_column = st.selectbox("Select Y variable:", df.columns)

    X = df[X_columns]
    Y = df[Y_column]
    
    # Train model
    st.subheader("Train Model")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train models with LazyPredict
    if st.button('Run LazyPredict'):
        if Y.dtypes == 'object':  # Classification task
            clf = LazyClassifier(predictions=True)
            models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)
            st.write(models)
            st.subheader('Evaluation Metrics for Best Model')
            st.write(predictions)
        else:  # Regression task
            reg = LazyRegressor(predictions=True)
            models, predictions = reg.fit(X_train, X_test, Y_train, Y_test)
            st.write(models)
            st.subheader('Evaluation Metrics for Best Model')
            st.write(predictions)
