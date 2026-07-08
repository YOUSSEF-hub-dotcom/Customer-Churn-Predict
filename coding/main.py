"""
Telco Customer Churn Prediction Project - Orchestration Pipeline.
Author: Youssef Mahmoud
Description: Central execution hub that coordinates data engineering, automated EDA, 
             model optimization using CatBoost, and performance tracking via MLflow.
"""

from data_pipeline import run_data_pipeline
from basic_eda import basic_eda
from advanced_eda import advanced_eda
from model import build_and_train_model
from MLflow_LifeCycle import run_mlflow_tracking
import argparse
import logging
from logger_config import setup_logging

# Initialize and configure application-wide production logging infrastructure
setup_logging()
logger = logging.getLogger("main")


def main():
    """
    Main orchestrator that parses execution arguments, loads configuration boundaries,
    and sequentially triggers the end-to-end Machine Learning lifestyle pipeline steps.
    """
    # Configure and capture execution hyperparameter inputs from CLI/Terminal
    parser = argparse.ArgumentParser(description="Production-Grade Churn Pipeline Execution Interface")
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum boosting iterations for CatBoost model")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="Optimization step size reduction factor")
    parser.add_argument("--depth", type=int, default=6, help="Maximum tree depth structure constraint")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("       Telco Customer Churn Prediction Production Pipeline       ")
    logger.info("=" * 60)

    # Define strict source path location for structural client demographic telemetry
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # --------------------------------------------------------------------------
    # STEP 1: INBOUND DATA ENGINEERING & ETL PIPELINE
    # --------------------------------------------------------------------------
    logger.info("\n>>> Step 1: Running Data Pipeline (Loading + Cleaning + Preprocessing)")
    df = run_data_pipeline(FILE_PATH)
    logger.info("Data Pipeline Ingestion and Preprocessing Completed Successfully!\n")

    # --------------------------------------------------------------------------
    # STEP 2: STATISTICAL EDA COHORT DISCOVERY
    # --------------------------------------------------------------------------
    logger.info(">>> Step 2: Running Basic Exploratory Data Analysis Profiles")
    basic_eda(df)

    # --------------------------------------------------------------------------
    # STEP 3: ADVANCED STRUCTURAL ATTRIBUTE CORRELATIONS
    # --------------------------------------------------------------------------
    logger.info(">>> Step 3: Running Advanced Exploratory Data Analysis & Churn Insights")
    advanced_eda(df)

    # --------------------------------------------------------------------------
    # STEP 4: MODEL DEVELOPMENT, TUNING, & COMPILATION
    # --------------------------------------------------------------------------
    logger.info(">>> Step 4: Compiling and Optimizing Production CatBoost Engine")
    model_results = build_and_train_model(
        df,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth
    )

    # --------------------------------------------------------------------------
    # STEP 5: AUDITING, LOGGING, & MLFLOW GOVERNANCE LIFECYCLE
    # --------------------------------------------------------------------------
    logger.info(">>> Step 5: Serializing Artifacts & Logging System Metadata to MLflow")
    run_mlflow_tracking(model_results)

    logger.info("=" * 60)
    logger.info("        End-To-End Operational Pipeline Completed Successfully!        ")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
