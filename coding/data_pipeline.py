import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Initialize logger for tracking pipeline execution stages
logger = logging.getLogger("Data_Pipeline")

def load_data(file_path):
    """
    Loads the dataset from the given CSV file path.
    """
    logger.info("Loading Dataset")
    df = pd.read_csv(file_path)
    return df


def basic_data_overview(df):
    """
    Performs an initial exploratory data analysis (EDA) and prints dataset metadata.
    """
    pd.set_option('display.width', None)
    
    logger.info("============ Initial 20 Rows Preview ============")
    print(df.head(20))

    logger.info("============ Basic Data Functions ============")
    logger.info("Information About Data:")
    df.info()  # Direct call, as df.info() handles printing internally

    logger.info("Statistical operations:")
    print(df.describe().round(2))

    logger.info("Rows & Columns of Data:")
    print(df.shape)

    logger.info("Columns of Data:")
    print(df.columns)

    logger.info("Data Types:")
    print(df.dtypes)

    logger.info("Display Index Range:")
    print(df.index)


def clean_and_preprocess_data(df):
    """
    Cleans missing values, standardizes categories, and handles feature engineering.
    """
    logger.info("============ Cleaning Data ============")

    logger.info("Number of duplicate rows:")
    logger.info(df.duplicated().sum()) 

    logger.info("Number of Missing Values:")
    print(df.isnull().sum()) 

    logger.info("============ Data Preprocessing ============")

    # Handle TotalCharges numerical conversion and impute missing values using MonthlyCharges
    logger.info('Converting TotalCharges to numeric and filling missing values...')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Ensure SeniorCitizen maintains correct integer standard representation
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Map predictable binary categories to numeric 0/1 for ML readiness
    logger.info('Convert all "Yes/No" values to 0/1')
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Standardize textual categories to eliminate redundant states
    logger.info('Replacing "No phone service" with "No" in MultipleLines...')
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    logger.info('Replacing "No internet service" with "No" in service columns...')
    replace_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No"})

    # Vectorized Feature Engineering: Calculate total active services efficiently without loops
    logger.info("============ Feature Engineering ============")
    logger.info("Creating 'NumServices' feature...")
    df['NumServices'] = (
        (df['PhoneService'] == 1).astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int)
    )

    # Logging final fully processed dataframe summary
    logger.info("Processed Dataset Preview:")
    print(df.head(20))

    logger.info("Final Data Types:")
    print(df.dtypes)

    return df


def run_data_pipeline(file_path):
    """
    Main orchestration function that runs the full sequential pipeline.
    """
    logger.info("============ Starting Data Pipeline ============")

    df = load_data(file_path)
    basic_data_overview(df)
    df_processed = clean_and_preprocess_data(df)

    logger.info("============ Data Pipeline Completed ============")

    return df_processed


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df_final = run_data_pipeline(FILE_PATH)
