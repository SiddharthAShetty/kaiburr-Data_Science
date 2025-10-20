# 🧩 Consumer Complaints Data Pipeline

## 📘 Overview
This project automates **data ingestion, cleaning, exploratory data analysis (EDA), feature extraction, model training, and prediction** for the **Consumer Financial Protection Bureau (CFPB) Consumer Complaints Dataset**.  
It is designed as an **end-to-end machine learning pipeline** that processes large CSV files in chunks, filters relevant complaint categories, and produces a **classification model** that predicts the complaint product type from textual narratives.

---

## 🧱 Key Features

### 🗃️ Data Ingestion & Filtering
- Reads large complaint CSV files in **chunks** to avoid memory issues.  
- Filters only relevant products:
  - `Credit reporting, credit repair services, or other personal consumer reports`
  - `Debt collection`
  - `Mortgage`
  - `Consumer Loan`
- Saves filtered data efficiently as a **compressed Parquet file** (`filtered_complaints.parquet`).

---

### 🔍 Exploratory Data Analysis (EDA)
Automatically generates an **EDA report** (`EDA_Model_Report.pdf`) with:
- Complaint length distribution  
- Product-wise complaint counts  
- Top sub-products, companies, and states  
- Wordclouds for each product  
- Top correlated unigrams and bigrams (via Chi-square test)  
- Model performance comparison (accuracy, F1-score, confusion matrix)

---

### 🧼 Text Preprocessing
Each complaint narrative is:
- Lowercased  
- Stripped of URLs, symbols, and punctuation  
- Tokenized and lemmatized  
- Stopwords removed  
- Ready for vectorization with **TF-IDF**

---

### 🧠 Machine Learning Models
Trains and compares three classifiers:
| Model | Description |
|--------|--------------|
| Logistic Regression | Baseline linear model for text classification |
| Random Forest | Ensemble of decision trees for nonlinear classification |
| XGBoost | Gradient-boosted trees with strong predictive power |

Each model is trained on a **TF-IDF feature matrix** and evaluated using:
- **Accuracy**
- **Macro F1-score**
- **Confusion matrix visualization**

The best model (highest F1-score) is automatically selected.

---

### 💾 Model Artifacts
All trained components are saved in the `artifacts/` folder:
- `tfidf.joblib` → TF-IDF vectorizer  
- `LogisticRegression.joblib`, `RandomForest.joblib`, `XGBoost.joblib` → trained models  

---

### 🧮 Prediction Function
After training, the script includes an **inference function** that:
1. Loads the trained model and TF-IDF vectorizer  
2. Cleans and vectorizes new complaint texts  
3. Predicts the most likely complaint product

## 📊 Outputs

| File / Folder | Description |
|----------------|-------------|
| `filtered_complaints.parquet` | Cleaned and filtered parquet dataset |
| `sample_for_model.parquet` | Balanced sampled dataset for modeling |
| `artifacts/` | Folder with saved TF-IDF and trained models |
| `EDA_Model_Report.pdf` | Full PDF report of visual EDA, model evaluation, and data insights |
| `Chi2_Feature_Analysis.pdf` | Detailed visualization of top correlated unigrams and bigrams per product based on Chi-square test |

**Example usage:**
```python
model, tfidf = load_artifacts("XGBoost")
texts = [
    "Debt collector keeps calling about a credit card I never took.",
    "I have trouble making my mortgage payments."
]
preds = predict_complaint(texts, model, tfidf)
print(preds)
