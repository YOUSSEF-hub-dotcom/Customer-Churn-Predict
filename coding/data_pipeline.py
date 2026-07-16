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
    logger.info("============ Cleaning Data & Preprocessing ============")

    logger.info("Number of duplicate rows:")
    logger.info(df.duplicated().sum()) 

    logger.info("Number of Missing Values:")
    print(df.isnull().sum()) 

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

    return df


def inspect_skew_and_outliers(df):
    """
    Performs a strict inspection on numerical features to discover Skewness 
    and Outlier distributions before the alignment/modelling firewall phase.
    NOTE: This is strictly for discovery and logging; actual mathematical transformations 
    must be executed inside the model's firewall to prevent Data Leakage.
    """
    logger.info("============ Advanced Inspection: Skew & Outliers ============")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    for col in numerical_cols:
        # Check Skewness
        current_skew = df[col].skew()
        
        # Check Outliers using IQR boundaries globally just for understanding
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        logger.info(f"Feature '{col}' -> Baseline Skewness: {current_skew:.4f} | Potential Outliers Count: {outliers_count}")

    # Logging final fully processed dataframe summary
    logger.info("Processed Dataset Preview:")
    print(df.head(20))

    logger.info("Final Data Types:")
    print(df.dtypes)


def analyze_spearman_correlations(df):
    """
    Computes and logs Spearman Rank Correlation to capture non-linear, monotonic
    relationships between key features and the target variable (Churn).
    """
    logger.info("============ Advanced Analysis: Spearman Correlation ============")
    
    # Selecting core continuous, engineered, and target features
    features_to_correlate = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'Churn']
    
    # Calculate the Spearman correlation matrix
    spearman_matrix = df[features_to_correlate].corr(method='spearman')
    
    # Print correlation values specifically relative to the target variable (Churn)
    logger.info("Spearman Correlation scores with respect to Churn:")
    print(spearman_matrix['Churn'].sort_values(ascending=False))
    
    # Note: Plt & Seaborn code for visual inspection can be run locally inside notebooks:
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Spearman Rank Correlation (Telco Churn)")
    # plt.show()


def run_data_pipeline(file_path):
    """
    Main orchestration function that runs the full sequential pipeline.
    """
    logger.info("============ Starting Data Pipeline ============")

    # 1. Load Data
    df = load_data(file_path)
    
    # 2. Basic Metadata Overview
    basic_data_overview(df)
    
    # 3. Global Structural Cleaning and Feature Engineering
    df_processed = clean_and_preprocess_data(df)
    
    # 4. Smart Check/Inspection for Outliers & Skewness on Cleaned Data
    inspect_skew_and_outliers(df_processed)

    # 5. Execute Spearman Rank Correlation Analysis
    analyze_spearman_correlations(df_processed)

    logger.info("============ Data Pipeline Completed ============")

    return df_processed


if __name__ == "__main__":
    # Setup baseline tracking logger output configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Define file path for loading dataset
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df_final = run_data_pipeline(FILE_PATH)
