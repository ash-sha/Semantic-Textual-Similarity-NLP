#%% md
# # 3.3.3 BioSentVec + Random Forest for Sentence Similarity
#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import sent2vec

#%%
data_path = "clinic.csv"
biosentvec_path = "/Users/aswath/PycharmProjects/mfac038/IndividualProject/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"

# Load dataset
df = pd.read_csv(data_path)
print(df.head())

# Split into train and test sets
train_text1 = df["Sent1"][:750].tolist()
train_text2 = df["Sent2"][:750].tolist()
train_labels = df["Score"][:750].tolist()

test_text1 = df["Sent1"][750:].tolist()
test_text2 = df["Sent2"][750:].tolist()
test_labels = df["Score"][750:].tolist()

#%%
model = sent2vec.Sent2vecModel()
try:
    model.load_model(biosentvec_path)
    print("BioSentVec model successfully loaded!")
except Exception as e:
    print(f"Error loading BioSentVec model: {e}")

#%%
def embed_sentences(sentences, model):
    """
    Generate embeddings for a list of sentences using BioSentVec.
    """
    return np.array([model.embed_sentence(sentence) for sentence in sentences])

#%%
train_emb1 = embed_sentences(train_text1, model)
train_emb2 = embed_sentences(train_text2, model)
test_emb1 = embed_sentences(test_text1, model)
test_emb2 = embed_sentences(test_text2, model)

print(f"Train Embedding Shapes: {train_emb1.shape}, {train_emb2.shape}")
print(f"Test Embedding Shapes: {test_emb1.shape}, {test_emb2.shape}")

#%%
reg1 = RandomForestRegressor(max_depth=6).fit(train_emb1, train_labels)
reg2 = RandomForestRegressor(max_depth=6).fit(train_emb2, train_labels)

#%%
test_pred1 = reg1.predict(test_emb1)
test_pred2 = reg2.predict(test_emb2)

mse1 = mean_squared_error(test_labels, test_pred1)
mse2 = mean_squared_error(test_labels, test_pred2)

print(f"MSE for Sentence 1: {mse1}")
print(f"MSE for Sentence 2: {mse2}")

#%%
def calculate_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity for two sets of embeddings.
    """
    return [1 - cosine(emb1[i], emb2[i]) for i in range(len(emb1))]

# Cosine Similarity for Train Data
train_cosine_sim = calculate_cosine_similarity(train_emb1, train_emb2)

# Cosine Similarity for Test Data
test_cosine_sim = calculate_cosine_similarity(test_emb1, test_emb2)

print(f"Sample Train Cosine Similarities: {train_cosine_sim[:5]}")
print(f"Sample Test Cosine Similarities: {test_cosine_sim[:5]}")

#%%
# Train Data
train_corr, _ = pearsonr(train_cosine_sim, train_labels)
print(f"Pearson Correlation for Train: {train_corr:.5f}")

# Test Data
test_corr, _ = pearsonr(test_cosine_sim, test_labels)
print(f"Pearson Correlation for Test: {test_corr:.5f}")

#%%
results = {
    "mse1": mse1,
    "mse2": mse2,
    "train_corr": train_corr,
    "test_corr": test_corr,
    "train_cosine_sim": train_cosine_sim,
    "test_cosine_sim": test_cosine_sim,
}
print("Results:", results)
