"""
Pelatihan Model Spam Detection dengan MLflow Project
Script ini digunakan untuk CI/CD workflow dengan GitHub Actions
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import os

def load_preprocessed_data():
    """
    Memuat dataset yang sudah dipreprocessing
    
    Returns:
        tuple: (X_tfidf, y)
    """
    # Load preprocessed data
    df = pd.read_csv('indo_spam_preprocessing.csv')
    
    # Load the vectorizer
    vectorizer = joblib.load('vectorizer.joblib')
    
    # Ambil cleaned text dan transform menggunakan vectorizer
    X_text = df['cleaned_text']
    X_tfidf = vectorizer.transform(X_text)
    
    # Ambil labels
    y = df['label']
    
    return X_tfidf, y

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Membuat dan menyimpan heatmap confusion matrix
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        save_path (str): Path untuk menyimpan plot
    
    Returns:
        str: Path file yang disimpan
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - Spam Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def save_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    """
    Menyimpan classification report detail
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        save_path (str): Path untuk menyimpan report
    
    Returns:
        str: Path file yang disimpan
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=['Ham', 'Spam'],
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPAM DETECTION - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    
    return save_path

def train_model(test_size=0.2, random_state=42, alpha=0.1):
    """
    Melatih model spam detection dengan MLflow tracking
    
    Args:
        test_size (float): Proporsi data test
        random_state (int): Random seed
        alpha (float): Alpha parameter untuk Multinomial NB
    """
    
    print("=" * 60)
    print("SPAM DETECTION - MLFLOW PROJECT TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Memuat data yang sudah dipreprocessing...")
    X, y = load_preprocessed_data()
    print(f"      Dataset loaded! {X.shape[0]} samples dengan {X.shape[1]} features")
    
    # Split data
    print("\n[2/5] Memisahkan data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # mlflow run automatically creates a run context, so we don't need to create one
    print("\n[3/5] Using MLflow run context...")
    
    # Train model with MLflow tracking
    print("\n[4/5] Melatih model...")
    
    # Log parameters
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    
    # Train model
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Hitung confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    mlflow.log_metric("true_negatives", int(tn))
    mlflow.log_metric("false_positives", int(fp))
    mlflow.log_metric("false_negatives", int(fn))
    mlflow.log_metric("true_positives", int(tp))
    mlflow.log_metric("specificity", tn / (tn + fp))
    
    print(f"\n      Performa Model:")
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-Score:  {f1:.4f}")
    print(f"      ROC-AUC:   {roc_auc:.4f}")
    
    # Buat artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    
    # Create artifacts
    print("\n[5/5] Membuat artifacts...")
    
    # Confusion matrix
    cm_path = plot_confusion_matrix(y_test, y_pred, 'artifacts/confusion_matrix.png')
    mlflow.log_artifact(cm_path)
    print("      Logged confusion matrix")
    
    # Classification report
    report_path = save_classification_report(y_test, y_pred, 
                                             'artifacts/classification_report.txt')
    mlflow.log_artifact(report_path)
    print("      Logged classification report")
    
    # Log trained model
    mlflow.sklearn.log_model(model, "model")
    print("      Logged trained model")
    
    # Simpan model lokal (untuk Docker)
    os.makedirs('models', exist_ok=True)
    model_path = 'models/spam_detection_model.joblib'
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    print(f"      Saved model to {model_path}")
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n      MLflow Run ID: {run_id}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return run_id

def main():
    """
    Main function dengan argument parsing untuk CLI
    """
    parser = argparse.ArgumentParser(description='Train spam detection model with MLflow')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter for Multinomial NB (default: 0.1)')
    
    args = parser.parse_args()
    
    print(f"\nTraining parameters:")
    print(f"  test_size: {args.test_size}")
    print(f"  random_state: {args.random_state}")
    print(f"  alpha: {args.alpha}\n")
    
    try:
        run_id = train_model(
            test_size=args.test_size,
            random_state=args.random_state,
            alpha=args.alpha
        )
        print(f"\nSuccess! MLflow run ID: {run_id}")
        return 0
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()
