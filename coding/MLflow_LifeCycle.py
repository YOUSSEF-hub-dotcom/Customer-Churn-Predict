import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger("MLflow")

def run_mlflow_tracking(results):
    """
    Synchronizes pipeline outputs with MLflow 2.x governance infrastructure. 
    Encapsulates dynamic tracking, visualization uploads, and Quality Gate promotions.
    """
    logger.info("--------------------------- Starting MLflow Tracking --------------------------")

    model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]
    best_thresh = results["best_threshold"]
    best_f1 = results["best_f1"]
    cm = results["confusion_matrix"]
    fi = results["feature_importance"]
    cv_auc_mean = results["cv_auc_mean"]
    cv_auc_std = results["cv_auc_std"]
    params = results["params"]

    X_COLUMNS = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
        'NumServices', 'TechSupport_OnlineSecurity'
    ]

    CATEGORICAL_FEATURES = [
        'Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'TechSupport_OnlineSecurity'
    ]

    # Calculate dynamic bounding limits from the current baseline run for deployment scaling
    outlier_bounds = {}
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        # Sqrt values for TotalCharges need to mirror the state used during training
        q1 = X_test[col].quantile(0.25)
        q3 = X_test[col].quantile(0.75)
        iqr = q3 - q1
        outlier_bounds[col] = {
            "lower": float(q1 - 1.5 * iqr),
            "upper": float(q3 + 1.5 * iqr)
        }

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Logging metrics: AUC={auc_score:.4f}, F1={best_f1:.4f}")

    scale_pos_weight = model.get_params()["scale_pos_weight"]

    mlflow.set_experiment("Churn_Prediction_CatBoost")
    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run(run_name="CatBoost_Production_Model") as run:

        # ---------------- Parameters ----------------
        mlflow.log_params(params)
        mlflow.log_param("best_threshold", float(best_thresh))
        mlflow.log_params({
            "l2_leaf_reg": model.get_params()["l2_leaf_reg"],
            "eval_metric": "AUC",
            "scale_pos_weight": float(scale_pos_weight),
            "early_stopping_rounds": model.get_params()["early_stopping_rounds"],
            "border_count": model.get_params()["border_count"],
            "bagging_temperature": model.get_params()["bagging_temperature"],
            "random_strength": model.get_params()["random_strength"],
        })

        # ---------------- Metrics ----------------
        mlflow.log_metrics({
            "accuracy": accuracy,
            "auc_score": auc_score,
            "best_f1_score": best_f1,
            "recall_churn": report_dict["1"]["recall"],
            "precision_churn": report_dict["1"]["precision"],
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std
        })

        # ---------------- Plots (Logged directly from memory) ----------------
        logger.info("Saving and logging visualization artifacts directly...")
        
        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed (0)", "Churn (1)"],
            yticklabels=["Stayed (0)", "Churn (1)"],
            ax=ax_cm
        )
        ax_cm.set_title("Confusion Matrix - Final Model")
        fig_cm.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # Feature Importance
        fig_fi, ax_fi = plt.subplots(figsize=(11, 8))
        sns.barplot(
            data=fi,
            x="Importances",
            y="Feature Id",
            hue="Feature Id",
            legend=False,
            ax=ax_fi
        )
        ax_fi.set_title("Top 15 Feature Importances - CatBoost")
        fig_fi.tight_layout()
        mlflow.log_figure(fig_fi, "feature_importance.png")
        plt.close(fig_fi)

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        fig_cc, ax_cc = plt.subplots(figsize=(7, 6))
        ax_cc.plot(prob_pred, prob_true, marker="o", label="CatBoost")
        ax_cc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_cc.set_title("Probability Calibration Curve")
        fig_cc.tight_layout()
        mlflow.log_figure(fig_cc, "calibration_curve.png")
        plt.close(fig_cc)

        # ---------------- JSON Artifacts ----------------
        logger.info("Exporting classification report and training info to MLflow Dashboard...")
        mlflow.log_dict(report_dict, "classification_report.json")
        mlflow.log_dict(fi.to_dict(orient="records"), "feature_importance.json")
        
        training_info = {
            "target": "Churn",
            "features": X_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES,
            "best_iteration": int(model.get_best_iteration()),
            "best_threshold": float(best_thresh),
            "scale_pos_weight": float(scale_pos_weight),
            "cv_auc_mean": cv_auc_mean,
            "cv_auc_std": cv_auc_std,
            "outlier_bounds": outlier_bounds
        }
        mlflow.log_dict(training_info, "training_info.json")

        # ---------------- Save Model ----------------
        joblib.dump(model, "catboost_model.pkl")
        mlflow.log_artifact("catboost_model.pkl")

        # ---------------- Production PyFunc Wrapper with Transformation Engine ----------------
        class ChurnPyFunc(mlflow.pyfunc.PythonModel):

            def __init__(self, threshold, feature_columns, bounds):
                self.threshold = threshold
                self.feature_columns = feature_columns
                self.bounds = bounds

            def load_context(self, context):
                self.model = joblib.load(context.artifacts["model"])

            def predict(self, context, model_input):
                if not isinstance(model_input, pd.DataFrame):
                    model_input = pd.DataFrame(model_input, columns=self.feature_columns)

                X = model_input[self.feature_columns].copy()
                
                # Dynamic Firewall: Apply inference outlier clipping mapping matching train baselines
                for col, limits in self.bounds.items():
                    if col in X.columns:
                        # If TotalCharges arrives raw at API, cap first based on training scale
                        X[col] = np.clip(X[col], limits["lower"], limits["upper"])
                
                # Mathematical Alignment: Apply identical Square Root transformation to TotalCharges
                if 'TotalCharges' in X.columns:
                    X['TotalCharges'] = np.sqrt(X['TotalCharges'])

                prob = self.model.predict_proba(X)[:, 1]
                pred = (prob >= self.threshold).astype(int)

                logger.info(f"Inference complete for {len(model_input)} records using decision boundary: {self.threshold}")

                return pd.DataFrame({
                    "churn_prediction": pred,
                    "churn_probability": prob
                })

        # ---------------- Signature ----------------
        input_example = X_test[X_COLUMNS].head(3)
        prob_example = model.predict_proba(input_example)[:, 1]
        pred_example = (prob_example >= best_thresh).astype(int)

        output_example = pd.DataFrame({
            "churn_prediction": pred_example,
            "churn_probability": prob_example
        })

        signature = infer_signature(input_example, output_example)

        # ---------------- Log PyFunc Model ----------------
        mlflow.pyfunc.log_model(
            artifact_path="churn_predictor_pyfunc",
            python_model=ChurnPyFunc(best_thresh, X_COLUMNS, outlier_bounds),
            artifacts={"model": "catboost_model.pkl"},
            input_example=input_example,
            signature=signature,
            registered_model_name="Churn_Predictor_PyFunc"
        )

        # ---------------- Registry & Promotion (MLflow 2.x standard) ----------------
        client = MlflowClient()
        model_name = "Churn_Predictor_PyFunc"
        run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/churn_predictor_pyfunc"
        registered_model = mlflow.register_model(model_uri, model_name)
        version = registered_model.version

        # Setting candidate alias as replacing the old staging status
        client.set_registered_model_alias(name=model_name, alias="candidate", version=version)

        # Quality Gate Verification Gateway
        if auc_score >= 0.80 and report_dict["1"]["recall"] >= 0.70:
            client.set_registered_model_alias(name=model_name, alias="champion", version=version)
            logger.info(
                f" Model version {version} passed Quality Gate and promoted to 'champion' (AUC: {auc_score:.2f}, Recall: {report_dict['1']['recall']:.2f})"
            )
        else:
            logger.warning(
                f" Quality Gate failed for version {version}. "
                f"Metrics: AUC={auc_score:.2f} (Req: 0.80), Recall={report_dict['1']['recall']:.2f} (Req: 0.70)"
            )

        logger.info(f" Model Governance Registry Update -> Name: {model_name} | Active Version: {version}")

        return run_id
