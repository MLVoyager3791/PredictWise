# PredictWise

This Streamlit application allows you to upload a CSV, TSV, or Excel file, clean the data, and automatically train multiple machine learning models using LazyPredict. The application will help you quickly evaluate different models for classification or regression tasks and display the results.

## Features
- Upload CSV, TSV, or Excel files for analysis
- Automatic data cleaning and handling of missing values
- Selection of feature variables (X) and target variable (Y)
- Automatic model training and evaluation using LazyPredict
- Display of evaluation metrics for the best models

## How to Use
1. **Upload File:**\
Click the "Browse files" button to upload your dataset in CSV, TSV, or Excel format.

2. **Data Cleaning:**\
The application will display the count of null values for each column.\
Choose how to handle missing values: Replace with Mean, Median, or Drop the rows with missing values.

3. **Select X and Y Variables:**\
Select the feature variables (X) and the target variable (Y) from the dropdown menus.

4. **Train Model:**\
The data will be split into training and testing sets.\
Click the "Run LazyPredict" button to automatically train multiple machine learning models.

5. **View Results:**\
The application will display the performance metrics for all the models trained.\
Detailed evaluation metrics for the best model will be shown.

## Requirements
- Python 3.6 or higher
- Streamlit
- Pandas
- LazyPredict
- Scikit-learn

## Installation

1. Clone the repository:
```
git clone https://github.com/MLVoyager3791/PredictWise.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Run the Streamlit application:
```
streamlit run app.py
```

## Example:

1. Upload a dataset file (CSV, TSV, or Excel).
2. Handle missing values by selecting an appropriate method (Mean, Median, or Drop).
3. Choose the feature and target variables.
4. Click "Run LazyPredict" to train the models.
5. View the evaluation metrics for the best models.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

⭐ Like what you see? Star the repo to help others discover it!
