import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import logging

# Initialize FastAPI app
app = FastAPI(title="Semantic Similarity API", description="API for Similarity analysis", version="1.0")

# Basic endpoint to check if the server is working
@app.get("/")
def read_root():
    return {"message": "Semantic Similarity API is running. Use /predict to analyze Similarity."}

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load tokenizer and model once at startup
MODEL_DIR = os.getenv('MODEL_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/')  # Default path if env var is not set

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define request schema
class SentencePair(BaseModel):
    sentence1: str
    sentence2: str

    @model_validator(mode='before')
    def check_sentences(cls, values):
        sentence1, sentence2 = values.get('sentence1'), values.get('sentence2')
        if not sentence1 or not sentence2:
            raise ValueError('Both sentences must be non-empty.')
        return values

@app.post("/predict/")
async def predict_similarity(payload: SentencePair):
    try:
        # Tokenize the input sentences
        inputs = tokenizer(
            payload.sentence1,
            payload.sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()  # Get similarity score

        logging.info(f"Predicted similarity score: {score}")
        return {"similarity_score": score}

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point to run the application
if __name__ == "__main__":
    # Run FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
