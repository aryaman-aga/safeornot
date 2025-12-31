# SafeOrNot: Hybrid Text Safety Detection System

A robust, hybrid text classification system designed to detect unsafe content, specifically tailored for the Indian context (Hinglish, slang, and subtle harassment).

## Features

- **Hybrid Detection Engine**: Combines a rule-based hard-filter for explicit profanity with a fine-tuned BERT model for contextual understanding.
- **Context Aware**: Distinguishes between safe activities (e.g., "walk and talk") and unsafe solicitations (e.g., "walk and have sex") even when sentence structures are identical.
- **Hinglish Support**: Trained on a synthetic dataset containing English, Hindi, and Hinglish (Indian English) examples.
- **FastAPI Server**: Production-ready API endpoint for real-time inference.

## Project Structure

```
.
├── generate_dataset.py    # Script to generate synthetic training data (10k samples)
├── train_model.py         # Script to fine-tune the BERT model
├── server.py              # FastAPI application for inference
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Dataset**:
    Creates a `dataset.csv` with 10,000 synthetic samples, including edge cases and hard negatives.
    ```bash
    python generate_dataset.py
    ```

3.  **Train Model**:
    Fine-tunes `prajjwal1/bert-small` on the generated dataset.
    ```bash
    python train_model.py
    ```
    The model will be saved to the `saved_model/` directory.

4.  **Run Server**:
    Starts the FastAPI server on port 8000.
    ```bash
    python server.py
    ```

## API Usage

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "text": "Today is a good day but you are nice, lets go for a walk and talk"
}
```

**Response**:
```json
{
  "text": "Today is a good day but you are nice, lets go for a walk and talk",
  "label": "safe",
  "confidence": 0.9999,
  "is_safe": true
}
```

## Hybrid Logic

The system first checks against a `HARD_CUSS_WORDS` list (in `server.py`) for immediate flagging of explicit content. If no explicit words are found, it passes the text to the BERT model for classification.

## Using the Pre-trained Model (Hugging Face)

You can use the pre-trained model directly from Hugging Face without retraining locally.

### 1. Python Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Replace with your Hugging Face model ID
model_id = "your-username/your-model-name" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "Today is a good day but you are nice, lets go for a walk and talk"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    print(probabilities)
```

### 2. Running the Server with HF Model
To run the server using the cloud model instead of a local checkpoint:

1. Open `server.py`.
2. Change `MODEL_PATH` to your Hugging Face Repo ID:
   ```python
   MODEL_PATH = "your-username/your-model-name"
   ```
3. Run the server:
   ```bash
   python server.py
   ```
