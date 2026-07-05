import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("Data_Pipeline")

def load_data(file_path):
    logger.info("Loading Dataset")
    df = pd.read_csv(file_path)
    return df


def basic_data_overview(df):
    pd.set_option('display.width', None)
    print(df.head(20))

    logger.info("============ Basic Functions ============")
    logger.info("Information About Data:")
    print(df.info())

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
    logger.info("============ Cleaning Data ============")

    logger.info("Number of duplicate rows:")
    logger.info(df.duplicated().sum()) # = 0

    logger.info("Number of Missing Values:")
    print(df.isnull().sum()) # = 0

    logger.info("============ Data Preprocessing ============")

    logger.info('Converting TotalCharges to numeric and filling missing values...')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    logger.info('Convert all "Yes/No" values to 0/1')
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    logger.info('Replacing "No phone service" with "No" in MultipleLines...')
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    logger.info('Replacing "No internet service" with "No" in service columns...')
    replace_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No internet service": "No"})

    logger.info("Processed Dataset Preview:")
    print(df.head(20))

    logger.info("Final Data Types:")
    print(df.dtypes)

    df['NumServices'] = (
        df[['PhoneService', 'InternetService', 'StreamingMovies', 'StreamingTV']]
        .apply(lambda x: sum([
            1 if x['PhoneService'] == 1 else 0,
            1 if x['InternetService'] != 'No' else 0,
            1 if x['StreamingMovies'] == 1 else 0,
            1 if x['StreamingTV'] == 1 else 0
        ]), axis=1)
    )

    return df


def run_data_pipeline(file_path):
    logger.info("============ Starting Data Pipeline ============")

    df = load_data(file_path)
    basic_data_overview(df)
    df_processed = clean_and_preprocess_data(df)

    logger.info("============ Data Pipeline Completed ============")

    return df_processed


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df_final = run_data_pipeline(FILE_PATH)
