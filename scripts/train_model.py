import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "model" / "saved_model"
RESULTS_DIR = BASE_DIR / "model" / "results"

# 1. Load Data
print(f"Loading dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Convert labels to integers
label_map = {"safe": 0, "not safe": 1}
df['label'] = df['label'].map(label_map)

# Split data
# FOR DEMO SPEED: Using only 500 samples. Remove [:500] for full training.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 2. Tokenization
# model_name = "distilbert-base-multilingual-cased" # Best for Hinglish (500MB+)
# model_name = "prajjwal1/bert-small" # Previous tiny model
model_name = "roberta-base" # Stronger English sense (OpenAI-style transformer architecture)
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

print("Tokenizing data...")
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

class SafeNotSafeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SafeNotSafeDataset(train_encodings, train_labels)
val_dataset = SafeNotSafeDataset(val_encodings, val_labels)

# 3. Model Setup
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(RESULTS_DIR / 'logs'),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

# 5. Save Model
# (Already saved above)

