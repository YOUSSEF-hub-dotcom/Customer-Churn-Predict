from data_pipeline import run_data_pipeline
from basic_eda import basic_eda
from advanced_eda import advanced_eda
from model import build_and_train_model
from MLflow_LifeCycle import run_mlflow_tracking
import argparse

import logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    logger.info("=" * 60)
    logger.info("       Telco Customer Churn Prediction Project       ")
    logger.info("=" * 60)


    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"

    logger.info("\n>>> Step 1: Running Data Pipeline (Loading + Cleaning + Preprocessing)")
    df = run_data_pipeline(FILE_PATH)
    logger.info("Data Pipeline Completed!\n")

    logger.info(">>> Step 2: Basic Exploratory Data Analysis")
    basic_eda(df)

    logger.info(">>> Step 3: Advanced Exploratory Data Analysis & Churn Insights")
    advanced_eda(df)

    logger.info(">>> Step 4: Building and Training CatBoost Model")
    model_results = build_and_train_model(
        df,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth
    )

    logger.info(">>> Step 5: Logging Everything to MLflow")
    run_mlflow_tracking(model_results)

    logger.info("=" * 60)
    logger.info("        Project Completed Successfully!       ")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
