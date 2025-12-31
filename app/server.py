from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import re

# Initialize FastAPI app
app = FastAPI(title="Safe/Not Safe Text Detection API")

# --- HYBRID SYSTEM: Hard Cuss Words List ---
# These words will trigger an immediate "not safe" response, bypassing the model.
# This ensures 100% catch rate for explicit profanity.
HARD_CUSS_WORDS = [
    # Hinglish / Hindi - Core Abuses
    "bsdk", "bhosdike", "bhosadike", "bhosdi", "bhosda",
    "mc", "madarchod", "maderchod", "maadarchod",
    "bc", "bhenchod", "behenchod", "benchod", "behen",
    "chutiya", "choot", "chu", "gandu", "gaandu", "gand", "gaand",
    "lodu", "laude", "lawde", "loda", "lund", "land",
    "randi", "chinnal", "kameena", "harami", "haramkhor",
    "tatte", "jhant", "jhaant",
    "chudai", "chodu", "randikhana", "rand", "madar",
    "suar", "kutta", "saale", "kamina", "kamine",
    "hijra", "chakka", "meetha", "betichod",
    "raand", "bhadwa", "bhadve", "bhadwe",
    "chut", "bur", "burr",
    
    # English Explicit
    "fuck", "fucker", "motherfucker", "fucking", "fucked",
    "bitch", "asshole", "bastard", "dick", "pussy", "cock",
    "whore", "slut", "cunt", "nigger", "nigga", "faggot", "retard",
    "rape", "rapist", "molest", "kill", "murder", "terrorist",
    "bomb", "shoot", "stab", "die", "suicide",
    "porn", "xxx", "nude", "naked", "sex", "boobs", "tits", "vagina", "penis"
]

def contains_hard_cuss_word(text: str):
    """
    Checks if the text contains any hard cuss words using regex word boundaries.
    Returns (bool, word_found).
    """
    text_lower = text.lower()
    for word in HARD_CUSS_WORDS:
        # Use regex to match whole words only (avoiding false positives like 'class' for 'ass')
        # \b matches word boundary
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            return True, word
    return False, None

# Load Model and Tokenizer
# Priority: Local '../model/saved_model' > Hugging Face 'aryaman1222/safe'
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = BASE_DIR / "model" / "saved_model"
HF_MODEL_ID = "aryaman1222/safe"

print("Loading model...")

try:
    # Try loading from local directory first (for development/testing)
    print(f"Attempting to load local model from '{LOCAL_MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    print("✅ Local model loaded successfully.")
except Exception as e_local:
    print(f"⚠️ Could not load local model: {e_local}")
    print(f"Attempting to load from Hugging Face Hub: {HF_MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
        print("✅ Hugging Face model loaded successfully.")
    except Exception as e_hf:
        print(f"\n❌ CRITICAL ERROR: Could not load model from local or Hugging Face.")
        print(f"Local Error: {e_local}")
        print(f"HF Error: {e_hf}")
        exit(1)

model.eval() # Set to evaluation mode

# Define Request Schema
class TextRequest(BaseModel):
    text: str

# Define Response Schema
class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_safe: bool

@app.get("/")
def read_root():
    return {"message": "Safe/Not Safe Text Detection API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    text = request.text
    
    # 1. Hybrid Check: Scan for Hard Cuss Words
    has_cuss_word, word_found = contains_hard_cuss_word(text)
    if has_cuss_word:
        print(f"Hybrid System: Detected hard cuss word '{word_found}'. Flagging as unsafe.")
        return {
            "text": text,
            "label": "not safe",
            "confidence": 1.0, # Absolute confidence
            "is_safe": False
        }
    
    # 2. Model Inference (if no hard cuss words found)
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
    # Get result
    # Label 0 = Safe, Label 1 = Not Safe
    safe_score = probabilities[0][0].item()
    unsafe_score = probabilities[0][1].item()
    
    if unsafe_score > safe_score:
        label = "not safe"
        confidence = unsafe_score
        is_safe = False
    else:
        label = "safe"
        confidence = safe_score
        is_safe = True
        
    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "is_safe": is_safe
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
