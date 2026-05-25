import spacy

# Load model bahasa (hanya dijalankan sekali saat file ini di-import)
print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text):
    """
    Data Preprocessing menggunakan spaCy.
    Meliputi: tokenisasi, hapus stop words, hapus tanda baca, dan lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Memproses teks melalui pipeline spaCy
    doc = nlp(text)
    
    # List comprehension untuk menyaring token
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num
    ]
    
    # Gabungkan kembali token yang sudah bersih menjadi sebuah string kalimat
    return " ".join(tokens)