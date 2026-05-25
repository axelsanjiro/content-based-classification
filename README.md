# Content-Based Digital Productivity Blocker API

An automated, machine learning-powered text classification REST API designed to enhance digital productivity. The system automatically filters and blocks distracting content (such as sports and entertainment) while allowing productive content (such as business, tech, and politics).

Built with **FastAPI** and **Scikit-learn**, utilizing **spaCy** for advanced natural language preprocessing and **Logistic Regression** as the core classification model based on the **BBC News Archive Dataset**.

## 🚀 Features
- **Advanced Text Preprocessing**: Powered by spaCy (`en_core_web_sm`) for state-of-the-art tokenization, stop-words removal, punctuation filtering, and lemmatization.
- **Context-Aware Feature Extraction**: Implements TF-IDF Vectorization optimized with word bigrams (`ngram_range=(1, 2)`) to capture phrasal context.
- **Production-Ready REST API**: Built using FastAPI for lightweight, high-performance execution and automatic interactive documentation generation (Swagger UI).
- **Kotlin Integration Friendly**: Tailored JSON request/response formats designed to easily hook into Android client applications for real-time interface blocking overlays.

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3.10+
- **Machine Learning Framework**: Scikit-learn
- **Natural Language Processing**: spaCy (Model: `en_core_web_sm`)
- **Web Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **Model Serialization**: Joblib
- **Data Manipulation**: Pandas & NumPy

## 📂 Project Structure
```text
content-based-classification/
│
├── dataset/
│   └── bbc-news-data.csv      # BBC News Archive Dataset (Tab-Separated)
│
├── models/
│   └── content_classifier.pkl # Trained Scikit-learn Pipeline (TF-IDF + Logistic Regression)
│
├── src/
│   ├── __init__.py            # Marks directory as a Python package
│   ├── preprocess.py          # Modular text cleaning pipeline using spaCy
│   └── train.py               # Model training, hyperparameter tuning, and evaluation script
│
├── app.py                     # FastAPI backend application setup
├── requirements.txt           # Python library dependencies
└── .gitignore                 # Excludes environments, datasets, and binaries from version control