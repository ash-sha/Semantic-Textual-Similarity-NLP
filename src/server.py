import os
import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(title="Semantic Similarity API", description="API for Similarity analysis", version="2.0")

# Load the vectorizer and Random Forest model at startup
VEC_DIR = os.getenv('VEC_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/3.Semantic-Textual-Similarity-NLP/models/vectorizer_model.pkl')
MODEL_DIR = os.getenv('MODEL_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/3.Semantic-Textual-Similarity-NLP/models/tfidf_rf_model.pkl')

# Load vectorizer and model once at app startup
try:
    vectorizer = joblib.load(VEC_DIR)
    rf_model = joblib.load(MODEL_DIR)
    logging.info(f"Successfully loaded model and vectorizer.")
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")
    raise Exception("Failed to load model or vectorizer.")

# Basic endpoint to check if the server is working
@app.get("/")
def read_root():
    return {"message": "Semantic Similarity API is running. Use /predict to analyze Similarity."}

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

# Function to calculate cosine similarity
def calculate_cosine_similarity(vecs1, vecs2):
    return np.array([cosine_similarity(vecs1[i], vecs2[i])[0][0] for i in range(vecs1.shape[0])])

# Inference function
def predict_sim(sentence1: str, sentence2: str):
    # Vectorize the input sentences using the loaded vectorizer
    vec1 = vectorizer.transform([sentence1])
    vec2 = vectorizer.transform([sentence2])

    # Calculate cosine similarity between the two sentences
    in_sim = calculate_cosine_similarity(vec1, vec2)

    # Combine features for prediction
    features = np.hstack([vec1.toarray(), vec2.toarray(), in_sim.reshape(-1, 1)])

    # Predict similarity score using the Random Forest model
    predicted_score = rf_model.predict(features)[0]

    return predicted_score

@app.post("/predict/")
async def predict_similarity(payload: SentencePair):
    try:
        # Call the predict_sim function to get the similarity score
        predicted_score = predict_sim(payload.sentence1, payload.sentence2)

        # Return the predicted score as JSON
        return {"similarity_score": predicted_score}

    except Exception as e:
        # Log any errors that occur during prediction
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Main entry point to run the application
if __name__ == "__main__":
    # Run FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
