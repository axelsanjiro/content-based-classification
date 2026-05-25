import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Import fungsi dari preprocess.py
from src.preprocess import preprocess_text_spacy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 1. UPDATE PATH: Ganti nama file dataset ke BBC News
DATA_PATH = os.path.join(ROOT_DIR, 'dataset', 'bbc-news-data.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'content_classifier.pkl')

def main():
    # 1. Data Collection
    print("Loading dataset...")
    # 2. UPDATE FORMAT BACA: Tambahkan sep='\t' karena dataset BBC dipisahkan oleh tab (bukan koma)
    df = pd.read_csv(DATA_PATH, sep='\t')
    
    # (Filter bahasa Inggris dihapus karena dataset BBC sudah murni berbahasa Inggris)
    
    # 3. UPDATE NAMA KOLOM: Hapus data kosong berdasarkan kolom asli di dataset BBC ('content' dan 'category')
    df = df.dropna(subset=['content', 'category'])
    
    # 2. Data Preprocessing (menggunakan spaCy)
    print("Preprocessing data with spaCy (this might take a moment)...")
    # 4. UPDATE TARGET FITUR: Gunakan kolom 'content'
    df['clean_text'] = df['content'].apply(preprocess_text_spacy)
    
    # Split Data (Fitur: clean_text, Target: category)
    # 5. UPDATE TARGET LABEL: Gunakan kolom 'category'
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['category'], test_size=0.2, random_state=42
    )
    
    # 3. Feature Extraction & 4. Model Training
    print("Extracting features and training models...")
    
    # Tuning Agresif dipertahankan (Akan memberikan performa sangat tinggi pada dataset BBC)
    lr_pipeline = Pipeline([
        (
            'tfidf', 
            TfidfVectorizer(
                ngram_range=(1, 2), # Membaca frasa (2 kata)
                max_df=0.9,         # Abaikan kata yang muncul di lebih dari 90% dokumen
                min_df=3            # Abaikan kata yang muncul kurang dari 3 kali
            )
        ),
        (
            'clf', 
            LogisticRegression(
                max_iter=2000, 
                class_weight='balanced', 
                C=10,                    
                random_state=42
            )
        )
    ])
    
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    lr_pipeline.fit(X_train, y_train)
    nb_pipeline.fit(X_train, y_train)
    
    # 5. Model Evaluation
    print("\n--- Evaluasi Logistic Regression ---")
    y_pred_lr = lr_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    
    print("\n--- Evaluasi Naive Bayes (Baseline) ---")
    y_pred_nb = nb_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_nb))
    
    # Simpan model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr_pipeline, MODEL_PATH)
    print(f"\nModel saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    main()