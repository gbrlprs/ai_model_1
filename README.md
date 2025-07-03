# Fake URL Detector CLI

A command-line tool for detecting potentially malicious URLs using machine learning. This project trains a Random Forest classifier on a labeled dataset of URLs, extracts interpretable features, and provides an easy-to-use CLI for training, prediction, and model interpretation.

## Features
- **Train** a machine learning model on a CSV dataset of URLs
- **Predict** whether a URL is malicious or benign
- **Batch prediction** from a file of URLs
- **Show feature importances** to interpret model decisions

## Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or manually install:
   ```sh
   pip install pandas scikit-learn joblib
   ```

## Usage

### 1. Train the Model
Train the model on your labeled CSV dataset:
```sh
python project2.py train --data path/to/malicious_phish.csv
```
- The model will be saved as `fake_url_detector.pkl` by default.

### 2. Predict a Single URL
Predict the class of a single URL:
```sh
python project2.py predict --url "http://example.com/login"
```

### 3. Predict Multiple URLs from a File
Predict classes for multiple URLs listed in a text file (one URL per line):
```sh
python project2.py predict --url_file urls.txt
```

### 4. Show Feature Importances
Display which features are most important to the model:
```sh
python project2.py feature_importances
```

### 5. Help
Show all available commands and options:
```sh
python project2.py -h
```

## Dataset
This project expects a CSV file with at least two columns:
- `url`: The URL string
- `type` or `label`: The class label (e.g., "benign", "phishing", etc.)

## Example Dataset Row
| url                        | type      |
|----------------------------|-----------|
| http://example.com/login   | phishing  |
| https://secure-bank.com    | benign    |

## Requirements
- Python 3.7+
- pandas
- scikit-learn
- joblib

## License
[MIT License](LICENSE)

---

**Disclaimer:** This tool is for educational and research purposes. Always validate results and do not use as the sole method for security-critical decisions.
