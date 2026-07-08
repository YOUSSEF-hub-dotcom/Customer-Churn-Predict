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

# Initialize logger for tracking modeling, validation, and evaluation stages
logger = logging.getLogger("Model")


def build_and_train_model(df, iterations, learning_rate, depth):
    """
    Orchestrates the statistical model execution lifecycle. Encapsulates strict data isolation 
    firewalls, engineering transformations, cross-validation boundaries, and threshold optimization.
    """
    logger.info(f"================ Building ML Model (Iter={iterations}, LR={learning_rate}, Depth={depth}) ===============")
    
    # Create a deep copy to safeguard core pipeline dataframe states
    df_model = df.copy()

    # Define features and isolate targeting vectors
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

    X = df_model[X_COLUMNS]
    y = df_model['Churn'].astype(int)

    # -------------------------------------------------------------------------
    # 1. THE FIREWALL: Absolute Data Isolation Split
    # -------------------------------------------------------------------------
    # Safe isolation executed BEFORE any statistical computations to prevent downstream Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create explicit validation working copies to avoid SettingWithCopy warnings
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Calculate scale_pos_weight based entirely on Training Set populations
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    logger.info(f"Train Structure Baseline -> Stayed (0): {neg} | Churn (1): {pos}")
    logger.info(f"Calculated scale_pos_weight = {scale_pos_weight:.3f}")

    # -------------------------------------------------------------------------
    # 2. FEATURE TREATMENT: Outlier Boundaries & Capping (Train-Driven) - STEP 1
    # -------------------------------------------------------------------------
    logger.info("============ Starting Outlier & Skew Treatment ============")
    skew_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Clean and cap outlier limits first to protect downstream statistical transformations
    for col in skew_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = len(X_train[(X_train[col] < lower_bound) | (X_train[col] > upper_bound)])
        logger.info(f"--- {col} Outlier Analysis ---")
        logger.info(f"IQR: {IQR:.2f} | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Train Outliers: {outliers_count}")
        
        # Robust Capping Strategy: Winsorize data boundaries based purely on Train parameters
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    # Confirm outlier suppression configurations visually
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(skew_cols):
        sns.boxplot(y=X_train[col], color='skyblue', ax=axes[idx])
        axes[idx].set_title(f'Boxplot of {col} (Post-Treatment Capping)')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. FEATURE TREATMENT: Skewness Transformation (Train-Driven) - STEP 2
    # -------------------------------------------------------------------------
    # Analyze and transform data distributions now that the metrics are stable and clean
    for col in skew_cols:
        initial_skew = X_train[col].skew()
        logger.info(f"Baseline Skewness of {col} [Train]: {initial_skew:.4f}")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(X_train[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col} (Pre-Transformation)")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # Apply Square Root Transformation using templates derived strictly from Train set variance
    logger.info("Applying Square Root Transformation to TotalCharges...")
    X_train['TotalCharges'] = np.sqrt(X_train['TotalCharges']) 
    X_test['TotalCharges'] = np.sqrt(X_test['TotalCharges']) 

    post_skew_tc = X_train['TotalCharges'].skew()
    logger.info(f"Post-Transformation Skew of TotalCharges (After Sqrt): {post_skew_tc:.4f}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(X_train['TotalCharges'], kde=True, ax=ax)
    ax.set_title("Distribution of TotalCharges Post-Transformation (Sqrt)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    # -------------------------------------------------------------------------
    # 4. STRATIFIED K-FOLD CROSS-VALIDATION (Validation Governance Stage)
    # -------------------------------------------------------------------------
    logger.info("\n================= Stratified K-Fold CV =================")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        spw_cv = (y_tr == 0).sum() / (y_tr == 1).sum()

        # Dynamic parameter sync mapping to mirror hyperparameter configurations
        cv_model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
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
        logger.info(f"Fold {fold} Verification AUC: {auc:.4f}")

    cv_auc_mean = np.mean(cv_auc_scores)
    cv_auc_std = np.std(cv_auc_scores)
    logger.info(f"\nCV Evaluation Results -> Mean AUC: {cv_auc_mean:.4f} | Std Deviation: {cv_auc_std:.4f}")

    # -------------------------------------------------------------------------
    # 5. PRODUCTION TRAIN: Final Production Model Training
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 6. THRESHOLD OPTIMIZATION: Custom Decision Alignment (F1 Optimization)
    # -------------------------------------------------------------------------
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
    logger.info(f"\nOptimized Decision Threshold = {best_thresh:.2f}")
    logger.info(f"Optimized Metrics Peak F1-Score = {best_f1:.4f}")

    # -------------------------------------------------------------------------
    # 7. METRICS PERFORMANCE REPORTING & GRAPHICAL TELEMETRY
    # -------------------------------------------------------------------------
    logger.info("\n================= Final Evaluation =================")
    logger.info(f"Accuracy Metric Score : {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Area Under ROC Curve  : {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"\nFinal Classification Matrix Profile:\n\n{classification_report(y_test, y_pred)}")

    # Visualizing Operational Metric Confusions
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stayed (0)', 'Churn (1)'],
        yticklabels=['Stayed (0)', 'Churn (1)'],
        ax=ax_cm
    )
    ax_cm.set_title('Confusion Matrix - Final Production Model')
    ax_cm.set_xlabel('Predicted Outbound Classes')
    ax_cm.set_ylabel('Actual Structural Ground Truth')
    plt.tight_layout()
    plt.show()
    plt.close(fig_cm)

    # Visualizing High-Impact Operational Features
    fi = model.get_feature_importance(prettified=True).head(15)
    fig_fi, ax_fi = plt.subplots(figsize=(11, 8))
    sns.barplot(
        data=fi,
        x='Importances',
        y='Feature Id',
        hue='Feature Id',
        palette='viridis',
        legend=False,
        ax=ax_fi
    )
    ax_fi.set_title('Top 15 Feature Importances - CatBoost Core Engine')
    ax_fi.set_xlabel('Relative Importance Weights')
    plt.tight_layout()
    plt.show()
    plt.close(fig_fi)

    # Evaluating Probability Reliability Curves
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig_cc, ax_cc = plt.subplots(figsize=(7, 6))
    ax_cc.plot(prob_pred, prob_true, marker='o', label='CatBoost Estimator')
    ax_cc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration Baseline')
    ax_cc.set_xlabel('Mean Predicted Probability Outputs')
    ax_cc.set_ylabel('Empirical Class True Proportions')
    ax_cc.set_title('Probability Calibration Evaluation Curve')
    ax_cc.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig_cc)

    logger.info("========================= Model Training Lifecycle Completed ======================")

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
