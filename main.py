import pandas as pd
import streamlit as st
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read file based on extension
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            return pd.read_csv(uploaded_file, sep='\t')
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Upload file
uploaded_file = st.file_uploader("Upload a CSV, TSV, or Excel file", type=["csv", "tsv", "xls", "xlsx"])

if uploaded_file is not None:
    # Read file
    df = read_file(uploaded_file)
    if df is not None:
        # Data cleaning
        st.subheader("Data Cleaning")
        null_counts = df.isnull().sum()
        st.write("Null counts:")
        st.write(null_counts)
        
        # Handling missing values
        if df.isnull().sum().sum() > 0:
            st.write("Choose how to handle missing values:")
            replace_option = st.radio("Replace with:", ('Mean', 'Median', 'Mode', 'Drop Rows', 'Drop Columns'))
            
            if replace_option == 'Mean':
                df = df.fillna(df.mean())
            elif replace_option == 'Median':
                df = df.fillna(df.median())
            elif replace_option == 'Mode':
                df = df.fillna(df.mode().iloc[0])
            elif replace_option == 'Drop Rows':
                df = df.dropna()
            elif replace_option == 'Drop Columns':
                df = df.dropna(axis=1)
        
        st.write("Cleaned Data:")
        st.write(df)
        
        # Data visualization
        st.subheader("Data Visualization")
        if st.checkbox("Show data distribution"):
            st.write(df.describe())
            for column in df.select_dtypes(include=['float64', 'int64']).columns:
                st.write(f"Distribution of {column}")
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

        if st.checkbox("Show Correlation Matrix"):
            st.write("Correlation Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Select X and Y variables
        st.subheader("Select X and Y Variables")
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        X_columns = st.multiselect("Select X variables:", df.columns)
        Y_column = st.selectbox("Select Y variable:", df.columns)

        if X_columns and Y_column:
            X = df[X_columns]
            Y = df[Y_column]
            
            st.write("Selected Features:")
            st.write(X.head())
            st.write("Selected Target:")
            st.write(Y.head())
        
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

                    # Visualization of predictions vs actual values
                    st.subheader("Predictions vs Actual")
                    fig, ax = plt.subplots()
                    ax.scatter(Y_test, predictions['predictions'], alpha=0.3)
                    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Predictions vs Actual')
                    st.pyplot(fig)
