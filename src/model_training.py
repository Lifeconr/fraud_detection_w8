import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.utils import load_data, save_data, logger

def prepare_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42) -> tuple:
    """Separate features and target, perform train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info(f"Prepared data: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train Logistic Regression model."""
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Trained Logistic Regression model")
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    model = lgb.LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, 
                               num_leaves=31, objective='binary', metric='aucpr',
                               scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    logger.info("Trained LightGBM model")
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, dataset_name: str) -> dict:
    """Evaluate model using AUC-PR, F1-Score, and Confusion Matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc_pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud (0)', 'Fraud (1)'], 
                yticklabels=['Not Fraud (0)', 'Fraud (1)'])
    plt.title(f'{dataset_name} - {model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f'data/processed/{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved {dataset_name} {model_name} confusion matrix to {cm_path}")
    
    return {'auc_pr': auc_pr, 'f1_score': f1, 'confusion_matrix': cm, 'cm_path': cm_path}

def interpret_model_shap(model, X_test: pd.DataFrame, model_name: str, dataset_name: str):
    logger.info(f"Generating SHAP plots for {dataset_name} {model_name}")
    
    # Initialize SHAP explainer
    if model_name == 'Logistic Regression':
        explainer = shap.LinearExplainer(model, X_test)
    else:
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1]
    else:
        shap_values_for_plot = shap_values

    # Use X_test directly
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_test, plot_type="bar", show=False)
    shap_bar_path = f'data/processed/{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}_shap_bar.png'
    plt.savefig(shap_bar_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_test, show=False)
    shap_dot_path = f'data/processed/{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}_shap_dot.png'
    plt.savefig(shap_dot_path)
    plt.close()

    logger.info(f"Saved SHAP plots: {shap_bar_path}, {shap_dot_path}")
    return {'shap_bar_path': shap_bar_path, 'shap_dot_path': shap_dot_path}

def main():
    """Main function to train and evaluate models."""
    # Load feature-engineered data
    fraud_df = load_data('data/processed/fraud_data_features.csv')
    creditcard_df = load_data('data/processed/creditcard_features.csv')
    
    # Define feature names (from Task 1 feature engineering)
    fraud_features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 
                      'transaction_count', 'avg_purchase_value', 'velocity'] + \
                     [col for col in fraud_df.columns if col.startswith(('source_', 'browser_', 'sex_', 'country_'))]
    creditcard_features = ['Time', 'Amount', 'hour_of_day'] + [f'V{i}' for i in range(1, 29)]
    
    # Prepare data
    fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test = prepare_data(fraud_df, 'class')
    creditcard_X_train, creditcard_X_test, creditcard_y_train, creditcard_y_test = prepare_data(creditcard_df, 'Class')
    
    # Train and evaluate models
    # Fraud Data
    fraud_lr_model = train_logistic_regression(fraud_X_train, fraud_y_train)
    fraud_lr_metrics = evaluate_model(fraud_lr_model, fraud_X_test, fraud_y_test, 'Logistic Regression', 'Fraud Data')
    fraud_lr_shap = interpret_model_shap(fraud_lr_model, fraud_X_test, 'Logistic Regression', 'Fraud Data', fraud_features)
    
    fraud_lgbm_model = train_lightgbm(fraud_X_train, fraud_y_train)
    fraud_lgbm_metrics = evaluate_model(fraud_lgbm_model, fraud_X_test, fraud_y_test, 'LightGBM', 'Fraud Data')
    fraud_lgbm_shap = interpret_model_shap(fraud_lgbm_model, fraud_X_test, 'LightGBM', 'Fraud Data', fraud_features)
    
    # Credit Card Data
    creditcard_lr_model = train_logistic_regression(creditcard_X_train, creditcard_y_train)
    creditcard_lr_metrics = evaluate_model(creditcard_lr_model, creditcard_X_test, creditcard_y_test, 'Logistic Regression', 'Credit Card')
    creditcard_lr_shap = interpret_model_shap(creditcard_lr_model, creditcard_X_test, 'Logistic Regression', 'Credit Card', creditcard_features)
    
    creditcard_lgbm_model = train_lightgbm(creditcard_X_train, creditcard_y_train)
    creditcard_lgbm_metrics = evaluate_model(creditcard_lgbm_model, creditcard_X_test, creditcard_y_test, 'LightGBM', 'Credit Card')
    creditcard_lgbm_shap = interpret_model_shap(creditcard_lgbm_model, creditcard_X_test, 'LightGBM', 'Credit Card', creditcard_features)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Dataset': ['Fraud Data', 'Fraud Data', 'Credit Card', 'Credit Card'],
        'Model': ['Logistic Regression', 'LightGBM', 'Logistic Regression', 'LightGBM'],
        'AUC-PR': [fraud_lr_metrics['auc_pr'], fraud_lgbm_metrics['auc_pr'], 
                   creditcard_lr_metrics['auc_pr'], creditcard_lgbm_metrics['auc_pr']],
        'F1-Score': [fraud_lr_metrics['f1_score'], fraud_lgbm_metrics['f1_score'], 
                     creditcard_lr_metrics['f1_score'], creditcard_lgbm_metrics['f1_score']],
        'Confusion Matrix Path': [fraud_lr_metrics['cm_path'], fraud_lgbm_metrics['cm_path'], 
                                 creditcard_lr_metrics['cm_path'], creditcard_lgbm_metrics['cm_path']],
        'SHAP Bar Plot Path': [fraud_lr_shap['shap_bar_path'], fraud_lgbm_shap['shap_bar_path'], 
                               creditcard_lr_shap['shap_bar_path'], creditcard_lgbm_shap['shap_bar_path']],
        'SHAP Dot Plot Path': [fraud_lr_shap['shap_dot_path'], fraud_lgbm_shap['shap_dot_path'], 
                               creditcard_lr_shap['shap_dot_path'], creditcard_lgbm_shap['shap_dot_path']]
    })
    save_data(metrics_df, 'data/processed/model_metrics.csv')
    logger.info("Saved model metrics to data/processed/model_metrics.csv")

if __name__ == "__main__":
    main()