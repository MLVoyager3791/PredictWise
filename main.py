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
        
        initial_count = len(df)
        
        # Handling missing values
        if df.isnull().sum().sum() > 0:
            st.write("Choose how to handle missing values:")
            replace_option = st.radio("Replace with:", ('Mean', 'Median', 'Mode', 'Drop Rows', 'Drop Columns'))
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
            
            if replace_option == 'Mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif replace_option == 'Median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif replace_option == 'Mode':
                df = df.fillna(df.mode().iloc[0])
            elif replace_option == 'Drop Rows':
                df = df.dropna()
            elif replace_option == 'Drop Columns':
                df = df.dropna(axis=1)
                
        final_count = len(df)
        if initial_count == final_count:
            st.markdown(f"### **Data is fully clean! Great job! 🎉**")
        else:
            st.markdown(f"### **Data cleaned!**\n\nBefore Cleaning: **{initial_count}**\n\nAfter Cleaning: **{final_count}**")
        
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
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                fig, ax = plt.subplots()
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.write("No numeric columns available for correlation matrix.")
        
        # Select X and Y variables
        st.subheader("Select X and Y Variables")
        X_columns = st.multiselect("Select X variables:", df.columns.tolist())
        auto_select_all = st.checkbox("Select all features as X")
        if auto_select_all:
            X_columns = list(df.columns)
            st.write("All features selected as X")

        y_column = st.selectbox("Select y variable:", df.columns)

        if X_columns and y_column:
            X = df[X_columns]
            y = df[y_column]
            
            st.write("Selected Features:")
            st.write(X.head())
            st.write("Selected Target:")
            st.write(y.head())
        
            # Train model
            st.subheader("Train Model")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models with LazyPredict
            if st.button('Run LazyPredict'):
                if y.dtypes == 'object':  # Classification task
                    clf = LazyClassifier(predictions=True)
                    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                    st.write(models)
                    st.subheader('Evaluation Metrics for Best Model')
                    st.write(predictions)

                else:  # Regression task
                    reg = LazyRegressor(predictions=True)
                    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                    st.write(models)
                    st.subheader('Evaluation Metrics for Best Model')
                    st.write(predictions)

                    # Debug: Inspect the structure of predictions DataFrame
                    st.write("Predictions DataFrame Structure:")
                    st.write(predictions.head())

                    # Assuming 'target' is the correct column name for predictions
                    st.subheader("Predictions vs Actual")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, predictions.iloc[:, 0], alpha=0.3)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Predictions vs Actual')
                    st.pyplot(fig)
