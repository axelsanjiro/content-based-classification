from fastapi import FastAPI
from pydantic import BaseModel
import joblib
# Mengimport fungsi pembersihan teks
from src.preprocess import preprocess_text_spacy

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Automated Content Classification API",
    description="API untuk memblokir konten digital guna meningkatkan produktivitas",
    version="1.0"
)

# Memuat pipeline model yang telah dilatih
model = joblib.load('models/content_classifier.pkl')

class ContentRequest(BaseModel):
    text: str

class ContentResponse(BaseModel):
    category: str
    action: str

@app.post("/predict", response_model=ContentResponse)
def predict_content(request: ContentRequest):
    # 1. Tahap Preprocessing
    cleaned_text = preprocess_text_spacy(request.text)
    
    # 2. Tahap Klasifikasi
    if not cleaned_text.strip():
        # Jika teks kosong, jangan blokir
        return ContentResponse(category="Unknown", action="allow")
        
    prediction = model.predict([cleaned_text])[0]
    
    # 3. Filtering System (Disesuaikan dengan kategori dataset BBC News)
    # Kategori produktif yang diizinkan (allow): business, tech, politics
    # Kategori distraksi yang diblokir (block): sport, entertainment
    blocked_categories = ["sport", "entertainment"]
    
    # Menentukan aksi overlay pemblokiran
    action = "block" if prediction.lower() in blocked_categories else "allow"
    
    return ContentResponse(
        category=prediction,
        action=action
    )