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

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Server**:
    The server will automatically download the pre-trained model from Hugging Face (`aryaman1222/safe`) and start the API.
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

## Development (Optional)

If you want to retrain the model from scratch:

1.  **Generate Dataset**:
    ```bash
    python generate_dataset.py
    ```

2.  **Train Model**:
    ```bash
    python train_model.py
    ```
