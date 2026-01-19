import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from data import load_and_split_data, get_preprocessing_pipeline
from sklearn.pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_baseline_model():
    logging.info("Starting model training process...")
    
    # 1. Load data
    try:
        X_train, X_test, y_train, y_test, num_cols, cat_cols = load_and_split_data()
        logging.info(f"Data loaded. Features: {list(num_cols)} (num), {list(cat_cols)} (cat)")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # 2. Build Pipeline
    preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)
    
    # Baseline: Logistic Regression (balanced to handle potential class imbalance)
    model = LogisticRegression(class_weight='balanced', random_state=42)
    
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 3. Train
    logging.info("Training Logistic Regression pipeline...")
    clf_pipeline.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
    
    logging.info("Model training complete. Evaluation Metrics:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    logging.info(f"ROC-AUC Score: {auc:.4f}")
    
    # 5. Save Model
    metadata = {
        'num_cols': list(num_cols),
        'cat_cols': list(cat_cols),
        'auc': auc,
        'algorithm': 'LogisticRegression'
    }
    
    joblib.dump(clf_pipeline, 'lead_scoring_model.pkl')
    joblib.dump(metadata, 'model_metadata.pkl')
    logging.info("Model saved as lead_scoring_model.pkl")
    
    return auc

if __name__ == "__main__":
    train_baseline_model()
