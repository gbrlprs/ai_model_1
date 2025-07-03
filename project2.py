import pandas as pd
import re
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

MODEL_FILENAME = 'fake_url_detector.pkl'

# Feature extraction function
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_https'] = int('https' in url)
    features['has_at'] = int('@' in url)
    features['has_ip'] = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url)))
    suspicious_words = ['login', 'verify', 'bank', 'free', 'secure', 'account']
    features['suspicious_words'] = sum(word in url.lower() for word in suspicious_words)
    return features

def train_model(data_path, model_path=MODEL_FILENAME):
    """Train the RandomForest model and save it to disk."""
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print("Columns in dataset:", data.columns)
    if 'type' in data.columns:
        y = data['type']
    elif 'label' in data.columns:
        y = data['label']
    else:
        print("Error: No 'type' or 'label' column found in dataset.")
        sys.exit(1)
    X = pd.DataFrame([extract_features(url) for url in data['url']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def predict_url(urls, model_path=MODEL_FILENAME):
    """Predict the class of one or more URLs using the trained model."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        sys.exit(1)
    model = joblib.load(model_path)
    features = pd.DataFrame([extract_features(url) for url in urls])
    preds = model.predict(features)
    for url, pred in zip(urls, preds):
        print(f"URL: {url}\nPrediction: {pred}\n")

def show_feature_importances(model_path=MODEL_FILENAME):
    """Display feature importances from the trained model."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        sys.exit(1)
    model = joblib.load(model_path)
    feature_names = ['url_length', 'num_dots', 'num_hyphens', 'has_https', 'has_at', 'has_ip', 'suspicious_words']
    importances = model.feature_importances_
    print("Feature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fake URL Detector CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model from a CSV dataset')
    train_parser.add_argument('--data', required=True, help='Path to CSV dataset')
    train_parser.add_argument('--model', default=MODEL_FILENAME, help='Path to save the trained model')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict the class of a URL or URLs')
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--url', help='A single URL to predict')
    group.add_argument('--url_file', help='A file containing URLs (one per line)')
    predict_parser.add_argument('--model', default=MODEL_FILENAME, help='Path to the trained model')

    # Feature importances command
    fi_parser = subparsers.add_parser('feature_importances', help='Show feature importances of the trained model')
    fi_parser.add_argument('--model', default=MODEL_FILENAME, help='Path to the trained model')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.data, args.model)
    elif args.command == 'predict':
        if args.url:
            urls = [args.url]
        else:
            with open(args.url_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        predict_url(urls, args.model)
    elif args.command == 'feature_importances':
        show_feature_importances(args.model)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
