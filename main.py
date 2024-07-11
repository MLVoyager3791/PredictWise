import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    if df.isnull().sum().sum() > 0: # Total number of null values in entire dataset
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
    X_columns = st.multiselect("Select X variables:", df.columns) # Only Numerical values as of now
    y_column = st.selectbox("Select Y variable:", df.columns)

    X = df[X_columns]
    y = df[y_column]
    
    # Train model
    st.subheader("Train Model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    
    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    accuracy = accuracy_score(y_test, Y_pred)
    st.write(f"Accuracy: {accuracy}")
    
    st.subheader("Classification Report")
    report = classification_report(y_test, Y_pred)
    st.text(report)
