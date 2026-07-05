import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
import logging

logger = logging.getLogger("MLflow")

def run_mlflow_tracking(results):

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

        # ---------------- Plots ----------------
        logger.info("Saving and logging visualization artifacts (Confusion Matrix, Feature Importance)...")
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed (0)", "Churn (1)"],
            yticklabels=["Stayed (0)", "Churn (1)"]
        )
        plt.title("Confusion Matrix - Final Model")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        plt.figure(figsize=(11, 8))
        sns.barplot(
            data=fi,
            x="Importances",
            y="Feature Id",
            hue="Feature Id",
            legend=False
        )
        plt.title("Top 15 Feature Importances - CatBoost")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

        plt.figure(figsize=(7, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="CatBoost")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("Probability Calibration Curve")
        plt.tight_layout()
        plt.savefig("calibration_curve.png", dpi=150)
        mlflow.log_artifact("calibration_curve.png")
        plt.close()

        # ---------------- JSON Artifacts ----------------
        logger.info("Exporting classification report and training info to JSON...")

        with open("classification_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        with open("feature_importance.json", "w") as f:
            json.dump(fi.to_dict(orient="records"), f, indent=2)
        mlflow.log_artifact("feature_importance.json")

        with open("training_info.json", "w") as f:
            json.dump({
                "target": "Churn",
                "features": X_COLUMNS,
                "categorical_features": CATEGORICAL_FEATURES,
                "best_iteration": int(model.get_best_iteration()),
                "best_threshold": float(best_thresh),
                "scale_pos_weight": float(scale_pos_weight),
                "cv_auc_mean": cv_auc_mean,
                "cv_auc_std": cv_auc_std
            }, f, indent=2)
        mlflow.log_artifact("training_info.json")

        # ---------------- Save Model ----------------
        joblib.dump(model, "catboost_model.pkl")
        mlflow.log_artifact("catboost_model.pkl")

        # ---------------- PyFunc Wrapper ----------------
        class ChurnPyFunc(mlflow.pyfunc.PythonModel):

            def __init__(self, threshold, feature_columns):
                self.threshold = threshold
                self.feature_columns = feature_columns

            def load_context(self, context):
                self.model = joblib.load(context.artifacts["model"])

            def predict(self, context, model_input):
                if not isinstance(model_input, pd.DataFrame):
                    model_input = pd.DataFrame(model_input, columns=self.feature_columns)

                X = model_input[self.feature_columns]
                prob = self.model.predict_proba(X)[:, 1]
                pred = (prob >= self.threshold).astype(int)


                logger.info(f"Generated predictions for {len(model_input)} records using threshold {self.threshold}")

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

        # ---------------- Log PyFunc ----------------
        mlflow.pyfunc.log_model(
            artifact_path="churn_predictor_pyfunc",
            python_model=ChurnPyFunc(best_thresh, X_COLUMNS),
            artifacts={"model": "catboost_model.pkl"},
            input_example=input_example,
            signature=signature,
            registered_model_name="Churn_Predictor_PyFunc"
        )

        # ---------------- Registry & Promotion ----------------
        client = MlflowClient()
        model_name = "Churn_Predictor_PyFunc"
        run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/churn_predictor_pyfunc"
        registered_model = mlflow.register_model(model_uri, model_name)
        version = registered_model.version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )

        if auc_score >= 0.80 and report_dict["1"]["recall"] >= 0.70:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(
                f" Model version {version} passed Quality Gate (AUC: {auc_score:.2f}, Recall: {report_dict['1']['recall']:.2f})")
        else:
            logger.warning(
                f" Quality Gate failed for version {version}. "
                f"Metrics: AUC={auc_score:.2f} (Req: 0.80), Recall={report_dict['1']['recall']:.2f} (Req: 0.70)"
            )

        logger.info(f" Model: {model_name} | Version: {version}")

        return run_id
