# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import sent2vec  # BioSentVec or other model for sentence embeddings

# Load the BioSentVec model (or use another model like Word2Vec)
model_path = "/path/to/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"  # Adjust the path to your model
model = sent2vec.Sent2vecModel()
model.load_model(model_path)

app = FastAPI()


# Pydantic model for input data
class SentencePair(BaseModel):
    sentence1: str
    sentence2: str


@app.post("/similarity")
def compute_similarity(pair: SentencePair):
    # Embed the sentences using BioSentVec (or any other model)
    embedding1 = model.embed_sentence(pair.sentence1)
    embedding2 = model.embed_sentence(pair.sentence2)

    # Compute cosine similarity
    from scipy.spatial.distance import cosine
    similarity = 1 - cosine(embedding1, embedding2)

    return {"similarity_score": similarity}

