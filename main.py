import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Upload file
uploaded_file = st.file_uploader("Upload a CSV, TSV, or Excel file", type=["csv", "tsv", "xls", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep='\t')
    else:
        df = pd.read_excel(uploaded_file)

    # Data cleaning
    st.subheader("Data Cleaning")
    null_counts = df.isnull().sum()
    st.write("Null counts:")
    st.write(null_counts)

    # Handling missing values
    if df.isnull().sum().sum() > 0:
        st.write("Choose how to handle missing values:")
        replace_option = st.radio("Replace with:", ['Mean', 'Median', 'Drop', 'Forward Fill', 'Backward Fill'])

        if replace_option == 'Mean':
            df.fillna(df.mean(), inplace=True)
        elif replace_option == 'Median':
            df.fillna(df.median(), inplace=True)
        elif replace_option == 'Drop':
            df.dropna(inplace=True)
        elif replace_option == 'Forward Fill':
            df.fillna(method='ffill', inplace=True)
        elif replace_option == 'Backward Fill':
            df.fillna(method='bfill', inplace=True)

    # Data visualization
    st.subheader("Data Visualization")
    if st.checkbox("Show data distribution"):
        st.write(df.describe())
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            st.write(f"Distribution of {column}")
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

    # Select X and Y variables
    st.subheader("Select X and Y Variables")
    X_columns = st.multiselect("Select X variables:", df.columns)
    y_column = st.selectbox("Select Y variable:", df.columns)

    if X_columns and y_column:
        X = df[X_columns]
        y = df[y_column]

        # Feature scaling
        st.subheader("Feature Scaling")
        scale_option = st.radio("Scale features?", ['No', 'Standard Scaling'])
        if scale_option == 'Standard Scaling':
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Train model
        st.subheader("Train Model")
        model_option = st.selectbox("Choose model:", ['Random Forest', 'Logistic Regression', 'Support Vector Machine'])
        if model_option == 'Random Forest':
            model = RandomForestClassifier()
        elif model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Support Vector Machine':
            model = SVC()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("Train"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation metrics
            st.subheader("Evaluation Metrics")
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

            # Hyperparameter tuning (optional)
            if st.checkbox("Perform Hyperparameter Tuning"):
                if model_option == 'Random Forest':
                    param_grid = {'n_estimators': [50, 100, 200]}
                elif model_option == 'Logistic Regression':
                    param_grid = {'C': [0.1, 1, 10]}
                elif model_option == 'Support Vector Machine':
                    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
                
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                st.write("Best Parameters:")
                st.write(grid_search.best_params_)

                y_pred_tuned = grid_search.predict(X_test)
                accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
                st.write(f"Tuned Accuracy: {accuracy_tuned}")

                st.subheader("Tuned Classification Report")
                report_tuned = classification_report(y_test, y_pred_tuned)
                st.text(report_tuned)
