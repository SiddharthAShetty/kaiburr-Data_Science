# full_consumer_complaints_pipeline_fixed.py
# Run with: python full_consumer_complaints_pipeline_fixed.py
# Requirements:
# pip install pandas pyarrow nltk scikit-learn matplotlib seaborn xgboost joblib wordcloud

import os
import re
import json
import logging
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import chi2

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# ------------------------------
# CONFIGURATION
# ------------------------------
FILE_PATH = r"C:\Users\HP\Downloads\complaints.csv"  # Change to your CSV
OUT_DIR = r"G:\kaiburr-task-solution\ds\processed"
os.makedirs(OUT_DIR, exist_ok=True)
PDF_REPORT_PATH = os.path.join(OUT_DIR, "EDA_Model_Report.pdf")
PARQUET_PATH = os.path.join(OUT_DIR, "filtered_complaints.parquet")
ARTIFACTS_DIR = os.path.join(OUT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CHUNKSIZE = 200_000
SAMPLE_MAX = 80_000
RANDOM_STATE = 42

# Products to keep
VALID_PRODUCTS = [
    'Credit reporting, credit repair services, or other personal consumer reports',
    'Debt collection',
    'Mortgage',
    'Consumer Loan'
]

# Sub-products per product; None means keep all
valid_filters = {
    'Credit reporting, credit repair services, or other personal consumer reports': None,
    'Debt collection': None,
    'Mortgage': None,
    'Consumer Loan': None
}

USECOLS = ['Product', 'Sub-product', 'Consumer complaint narrative', 'Date received', 'Company', 'State', 'ZIP code']

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ------------------------------
# NLTK safe downloads
# ------------------------------
for pkg in ('stopwords', 'wordnet', 'omw-1.4'):
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        logging.info(f"NLTK package '{pkg}' not found. Downloading...")
        nltk.download(pkg)

stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------------------
# Helper functions
# ------------------------------
def clean_text(text: str) -> str:
    """Basic cleaning + lemmatization. Uses NLTK stopwords defined above."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop and len(w) > 1]
    return " ".join(words)

# ------------------------------
# 1) FILTERING & PARQUET SAVE (safe writer)
# ------------------------------

def filter_and_save_parquet(csv_path, parquet_path):
    if os.path.exists(parquet_path):
        logging.info(f"Filtered parquet already exists at {parquet_path}")
        return

    total_kept = 0
    chunks_processed = 0
    writer = None

    try:
        for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=CHUNKSIZE, dtype=str, low_memory=True):
            chunks_processed += 1
            # drop missing complaints
            chunk = chunk[~chunk['Consumer complaint narrative'].isna()].copy()
            chunk['Product'] = chunk['Product'].str.strip()
            chunk['Sub-product'] = chunk['Sub-product'].fillna('').str.strip()

            # Filter by product and subproduct
            cand = chunk[chunk['Product'].isin(VALID_PRODUCTS)]
            if cand.empty:
                continue

            cand = cand[cand.apply(
                lambda row: True if valid_filters.get(row['Product']) is None
                else row['Sub-product'] in valid_filters.get(row['Product'], []),
                axis=1
            )]
            if cand.empty:
                continue

            # Convert to Arrow Table
            table = pa.Table.from_pandas(cand)

            # Append instead of overwrite
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
            writer.write_table(table)

            total_kept += len(cand)
            logging.info(f"[CHUNK {chunks_processed}] kept {len(cand)} rows; total_kept={total_kept}")
    finally:
        if writer is not None:
            writer.close()

    logging.info(f"[DONE] Filtered total rows written: {total_kept}")


# ------------------------------
# Main processing pipeline
# ------------------------------

def main(csv_path=FILE_PATH):
    # 1) filter and save parquet
    filter_and_save_parquet(csv_path, PARQUET_PATH)

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Expected parquet at {PARQUET_PATH} but not found")

    # 2) load filtered data
    df = pd.read_parquet(PARQUET_PATH)
    logging.info("Filtered dataframe shape: %s", df.shape)

    # 3) exploratory data analysis -> write figures into PDF (safe close)
    pdf = PdfPages(PDF_REPORT_PATH)
    try:
        # Complaint length
        df['text_len_chars'] = df['Consumer complaint narrative'].str.len()
        df['text_len_words'] = df['Consumer complaint narrative'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)

        plt.rcParams.update({
            'figure.figsize': (11.69, 8.27),  # A4 landscape
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.5,
            'font.size': 12
        })

        plt.figure(figsize=(6, 4))
        sns.histplot(df['text_len_words'], bins=50)
        plt.title("Distribution of Complaint Length (words)")
        plt.xlabel("Words")
        plt.ylabel("Count")
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Product distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(y='Product', data=df, order=df['Product'].value_counts().index)
        plt.title("Complaints per Product")
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Top Sub-products per Product
        for product in VALID_PRODUCTS:
            sub_df = df[df['Product'] == product]
            if sub_df.empty:
                continue
            top_subs = sub_df['Sub-product'].value_counts().head(10)
            plt.figure(figsize=(6, 4))
            sns.barplot(x=top_subs.values, y=top_subs.index)
            plt.title(f"Top Sub-products for {product}")
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Top companies
        plt.figure(figsize=(6, 4))
        top_comp = df['Company'].value_counts().head(10)
        sns.barplot(x=top_comp.values, y=top_comp.index)
        plt.title("Top Companies by Complaints")
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Top states
        plt.figure(figsize=(6, 4))
        top_states = df['State'].value_counts().head(10)
        sns.barplot(x=top_states.values, y=top_states.index)
        plt.title("Top States by Complaints")
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Wordcloud per product
        for product in VALID_PRODUCTS:
            texts = df[df['Product'] == product]['Consumer complaint narrative'].astype(str).apply(clean_text)
            combined_text = " ".join(texts)
            if len(combined_text.strip()) == 0:
                continue
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Wordcloud for {product}")
            pdf.savefig(bbox_inches='tight')
            plt.close()

        # Save some individual PNGs as well for quick inspection
        try:
            plt.figure(figsize=(6, 4))
            sns.countplot(y='Product', data=df, order=df['Product'].value_counts().index)
            plt.title("Complaints per Product")
            png_path = os.path.join(ARTIFACTS_DIR, 'complaints_per_product.png')
            plt.savefig(png_path, bbox_inches='tight')
            plt.close()
        except Exception:
            logging.warning('Failed to save quick PNGs — continuing')

    finally:
        pdf.close()
        logging.info(f"[DONE] EDA and model report saved at {PDF_REPORT_PATH}")

    # 4) label encoding & stratified sample
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Product'])

    counts = df['label'].value_counts().to_dict()
    n_classes = len(counts)
    per_class = max(10, SAMPLE_MAX // n_classes)
    sample_frames = []
    for lbl, cnt in counts.items():
        sub = df[df['label'] == lbl]
        n = min(per_class, cnt)
        if n <= 0:
            continue
        sample_frames.append(sub.sample(n=n, random_state=RANDOM_STATE))
    sample_df = pd.concat(sample_frames).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    sample_path = os.path.join(OUT_DIR, "sample_for_model.parquet")
    sample_df.to_parquet(sample_path, index=False)
    logging.info(f"Sample saved at {sample_path} (shape: %s)", sample_df.shape)

    # 5) text preprocessing on sample
    sample_df['clean_text'] = sample_df['Consumer complaint narrative'].apply(clean_text)

    # 6) feature extraction (TF-IDF)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, max_features=50000, ngram_range=(1, 2), stop_words='english')
    X = tfidf.fit_transform(sample_df['clean_text'])
    y = sample_df['label'].values

    joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf.joblib"))

    # 7) CHI-SQUARE analysis (most correlated unigrams/bigrams) — corrected alignment
    N = 5
    feature_names = np.array(tfidf.get_feature_names_out())
    category_to_id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    pdf_chi = PdfPages(os.path.join(OUT_DIR, "chi2_features.pdf"))
    try:
        for product, category_id in sorted(category_to_id.items()):
            chi2_vals, pvals = chi2(X, y == category_id)
            # descending order
            sorted_idx = np.argsort(chi2_vals)[::-1]

            # unigrams
            unigram_mask = np.array([1 if len(fn.split()) == 1 else 0 for fn in feature_names], dtype=bool)
            unigram_idx = [i for i in sorted_idx if unigram_mask[i]]
            top_unigrams_idx = unigram_idx[:N]
            top_unigrams = feature_names[top_unigrams_idx]
            top_unigrams_scores = chi2_vals[top_unigrams_idx]

            if len(top_unigrams) > 0:
                plt.figure(figsize=(6, 4))
                plt.barh(range(len(top_unigrams)), top_unigrams_scores[::-1], tick_label=top_unigrams[::-1])
                plt.title(f"Top {len(top_unigrams)} Correlated Unigrams for {product}")
                pdf_chi.savefig(bbox_inches='tight')
                plt.close()

            # bigrams
            bigram_mask = np.array([1 if len(fn.split()) == 2 else 0 for fn in feature_names], dtype=bool)
            bigram_idx = [i for i in sorted_idx if bigram_mask[i]]
            top_bigrams_idx = bigram_idx[:N]
            top_bigrams = feature_names[top_bigrams_idx]
            top_bigrams_scores = chi2_vals[top_bigrams_idx]

            if len(top_bigrams) > 0:
                plt.figure(figsize=(6, 4))
                plt.barh(range(len(top_bigrams)), top_bigrams_scores[::-1], tick_label=top_bigrams[::-1])
                plt.title(f"Top {len(top_bigrams)} Correlated Bigrams for {product}")
                pdf_chi.savefig(bbox_inches='tight')
                plt.close()
    finally:
        pdf_chi.close()
        logging.info("Saved chi2 feature plots at %s", os.path.join(OUT_DIR, "chi2_features.pdf"))

    # 8) train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 9) model training & comparison
    models = {
        "LogisticRegression": LogisticRegression(solver='saga', max_iter=200, C=1.0, n_jobs=-1, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
    }

    fitted_models = {}
    results = {}

    for name, model in models.items():
        logging.info(f"[TRAINING] {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        logging.info(f"{name} Accuracy={acc:.4f}, F1-macro={f1:.4f}")
        logging.info('\n' + classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        fitted_models[name] = model
        results[name] = {"accuracy": float(acc), "f1_macro": float(f1)}
        joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{name}.joblib"))

    # Model comparison plot
    res_df = pd.DataFrame(results).T
    plt.figure(figsize=(6, 4))
    sns.barplot(x=res_df.index, y=res_df['f1_macro'].values)
    plt.title("Model F1-macro Comparison")
    comp_png = os.path.join(ARTIFACTS_DIR, 'model_comparison.png')
    plt.savefig(comp_png, bbox_inches='tight')
    plt.close()

    # 10) best model confusion matrix
    best_name = res_df['f1_macro'].idxmax()
    best_model = fitted_models[best_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    cm_png = os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_png, bbox_inches='tight')
    plt.close()

    logging.info("Best model: %s", best_name)

    # 11) save metadata (label encoder + results)
    joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    metadata = {
        "best_model": best_name,
        "results": results,
        "label_classes": list(label_encoder.classes_)
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 12) prediction helpers
    def load_artifacts(model_name=None):
        if model_name is None:
            with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
                metadata_local = json.load(f)
            model_name_local = metadata_local.get("best_model")
        else:
            model_name_local = model_name

        model_obj = joblib.load(os.path.join(ARTIFACTS_DIR, f"{model_name_local}.joblib"))
        tfidf_obj = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf.joblib"))
        label_enc = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
        return model_obj, tfidf_obj, label_enc

    def predict_complaint(texts, model_obj, tfidf_obj, label_enc):
        cleaned = [clean_text(t) for t in texts]
        X_new = tfidf_obj.transform(cleaned)
        preds = model_obj.predict(X_new)
        return label_enc.inverse_transform(preds)

    # Example usage
    model_for_inference, tfidf_obj, label_enc = load_artifacts(best_name)
    examples = [
        "I found an error on my credit report and the bureau won't fix it.",
        "Debt collector keeps calling about a credit card I never took.",
        "I have trouble making my mortgage payments and the servicer won't help."
    ]
    preds = predict_complaint(examples, model_for_inference, tfidf_obj, label_enc)
    for t, p in zip(examples, preds):
        logging.info("TEXT: %s", t)
        logging.info("PREDICTED: %s", p)
        logging.info("%s", "-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full consumer complaints pipeline')
    parser.add_argument('--csv', type=str, default=FILE_PATH, help='Path to complaints CSV')
    args = parser.parse_args()
    main(args.csv)
