import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    data_path = os.path.join(PROCESSED_DATA_DIR, "model_features.csv")
    df = pd.read_csv(data_path)
    return df

def train_and_evaluate(df):
    logging.info("Preparing data for modeling...")
    
    # Drop rows with NaNs in strictly required columns just in case
    df = df.dropna(subset=['team1', 'team2', 'venue', 'city', 'target'])
    
    X = df[['team1', 'team2', 'venue', 'city', 'team1_won_toss', 'toss_decision_bat']]
    y = df['target']
    
    # Preprocessing
    categorical_features = ['team1', 'team2', 'venue', 'city']
    numeric_features = ['team1_won_toss', 'toss_decision_bat']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # In case of probability output
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        
        logging.info(f"{name} Results:")
        logging.info(f"Accuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f}")
        logging.info("\n" + classification_report(y_test, y_pred))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = clf
            best_name = name
            
    logging.info(f"Best Model is {best_name} with {best_accuracy:.4f} accuracy. Saving...")
    
    model_path = os.path.join(MODELS_DIR, "match_predictor.pkl")
    joblib.dump(best_model, model_path)
    
    # Save the columns to know inputs for predictions
    feature_cols = categorical_features + numeric_features
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    
    logging.info(f"Model saved at {model_path}.")

if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
