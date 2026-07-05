import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
from sklearn.calibration import calibration_curve
import logging
logger = logging.getLogger("Model")

def build_and_train_model(df, iterations, learning_rate, depth):
   logger.info(f"================ Building ML Model (Iter={iterations}, LR={learning_rate}, Depth={depth}) ===============")
   
    # Starting Skew & Outlier Treatment
    logger.info("============ Starting Skew & Outlier Treatment ============")
    
    # 1. Skewness Analysis Before Treatment
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in cols:
        skew_value = df[col].skew()
        logger.info(f"Skewness of {col} before treatment: {skew_value:.4f}")
        
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col} Before Treatment Skew")
        plt.show()

    # 2. Apply Square Root Transformation to handle Moderately Skewed TotalCharges
    logger.info("Applying Square Root Transformation to TotalCharges...")
    df['TotalCharges'] = np.sqrt(df['TotalCharges']) 

    treat_skew_TotalCharges = df['TotalCharges'].skew()
    logger.info(f"Treatment Skew of TotalCharges (After Sqrt): {treat_skew_TotalCharges:.4f}")
    
    sns.histplot(df['TotalCharges'], kde=True)
    plt.title("Distribution of TotalCharges After Treatment Skew (Sqrt)")
    plt.show()

    # 3. Outlier Detection using IQR Method
    print(df[cols].describe().round(2))

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        logger.info(f"--- {col} Outlier Analysis ---")
        logger.info(f"IQR: {IQR:.2f} | Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")
        logger.info(f"Number of Outliers Detected: {len(outliers)}\n")

    # 4. Plot Boxplots to confirm Outliers are handled
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cols, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f'Boxplot of {col} (Post-Treatment)')

    plt.tight_layout()
    plt.show()
    
    # feature Selection
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

    X = df[X_COLUMNS]
    y = df['Churn'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Handle Imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    logger.info(f"Train → Stayed (0): {neg} | Churn (1): {pos}")
    logger.info(f"scale_pos_weight = {scale_pos_weight:.3f}")

    logger.info("\n================= Stratified K-Fold CV =================")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        neg_cv = (y_tr == 0).sum()
        pos_cv = (y_tr == 1).sum()
        spw_cv = neg_cv / pos_cv

        cv_model = CatBoostClassifier(
            iterations=1500,
            depth=6,
            learning_rate=0.03,
            eval_metric='AUC',
            scale_pos_weight=spw_cv,
            random_seed=42,
            verbose=False
        )

        cv_model.fit(
            X_tr, y_tr,
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            use_best_model=True
        )

        val_prob = cv_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_prob)
        cv_auc_scores.append(auc)

        logger.info(f"Fold {fold} AUC: {auc:.4f}")

    cv_auc_mean = np.mean(cv_auc_scores)
    cv_auc_std = np.std(cv_auc_scores)

    logger.info(f"\nCV AUC Mean: {cv_auc_mean:.4f}")
    logger.info(f"CV AUC Std : {cv_auc_std:.4f}")

    logger.info("\n================= Training Final Model =================")

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='AUC',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=200,
        verbose=200,
        od_type='Iter',
        border_count=254,
        bagging_temperature=0.8,
        random_strength=1.0,
        task_type="CPU",
        thread_count=-1
    )

    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=False
    )

    # Threshold Optimization
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.35, 0.55, 0.01)
    best_f1, best_thresh, best_pred = 0, 0.5, None

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_pred = preds

    y_pred = best_pred

    logger.info(f"\nBest Threshold = {best_thresh:.2f}")
    logger.info(f"Best F1-Score = {best_f1:.4f}")

    logger.info("\n================= Final Evaluation =================")
    logger.info(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred))


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stayed (0)', 'Churn (1)'],
        yticklabels=['Stayed (0)', 'Churn (1)']
    )
    plt.title('Confusion Matrix - Final Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    fi = model.get_feature_importance(prettified=True).head(15)
    plt.figure(figsize=(11, 8))
    sns.barplot(
        data=fi,
        x='Importances',
        y='Feature Id',
        hue='Feature Id',
        palette='viridis',
        legend=False
    )
    plt.title('Top 15 Feature Importances - CatBoost')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    #  Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='CatBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    logger.info("========================= Model Training Completed ======================")

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'cv_auc_mean': cv_auc_mean,
        'cv_auc_std': cv_auc_std,
        'confusion_matrix': cm,
        'feature_importance': fi,
        'params': {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'scale_pos_weight': scale_pos_weight
        },
    }


if __name__ == "__main__":
    from data_pipeline import run_data_pipeline

    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = run_data_pipeline(FILE_PATH)

    results = build_and_train_model(df)
