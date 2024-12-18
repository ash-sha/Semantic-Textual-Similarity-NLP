import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load, Parallel, delayed
import warnings

warnings.filterwarnings('always')

# Load dataset and Word2Vec model
data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/clinic_c.csv"
word2vec_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/PubMed-and-PMC-w2v.bin"

# Load dataset
df = pd.read_csv(data_path)
print(df.head())

# Split data into train/test sets
train_text1 = df["Sent1"][:750].tolist()
train_text2 = df["Sent2"][:750].tolist()
train_labels = df["Score"][:750].tolist()

test_text1 = df["Sent1"][750:].tolist()
test_text2 = df["Sent2"][750:].tolist()
test_labels = df["Score"][750:].tolist()

# Load Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
print(f"Word2Vec Model Loaded with {len(word_vectors)} words.")

# Function to compute average feature vector for a sentence
def avg_feature_vector(sentence, model, num_features):
    """Compute the average feature vector for a sentence."""
    words = sentence.split()
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in model:
            n_words += 1
            feature_vec += model[word]
    if n_words > 0:
        feature_vec /= n_words
    return feature_vec

# Function to compute feature vectors for a list of sentences in parallel
def compute_vectors(sentences, model, num_features):
    """Compute feature vectors for a list of sentences in parallel."""
    return Parallel(n_jobs=-1)(
        delayed(avg_feature_vector)(sentence, model, num_features) for sentence in sentences
    )

# Compute feature vectors for train and test sets
train_vecs1 = compute_vectors(train_text1, word_vectors, 200)
train_vecs2 = compute_vectors(train_text2, word_vectors, 200)
test_vecs1 = compute_vectors(test_text1, word_vectors, 200)
test_vecs2 = compute_vectors(test_text2, word_vectors, 200)

print(f"Train Vecs1: {len(train_vecs1)}, Test Vecs1: {len(test_vecs1)}")

# Train Random Forest Regressors
reg1 = RandomForestRegressor(max_depth=6).fit(train_vecs1, train_labels)
reg2 = RandomForestRegressor(max_depth=6).fit(train_vecs2, train_labels)

# Save the trained models
model_path1 = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/model1w2v.joblib"
model_path2 = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/model2w2v.joblib"

dump(reg1, model_path1)
dump(reg2, model_path2)

print(f"Models saved to {model_path1} and {model_path2}")

# Predictions on the test set
test_pred1 = reg1.predict(test_vecs1)
test_pred2 = reg2.predict(test_vecs2)

# Mean Squared Error (MSE)
mse1 = mean_squared_error(test_labels, test_pred1)
mse2 = mean_squared_error(test_labels, test_pred2)
print(f"MSE for Sentence 1: {mse1}")
print(f"MSE for Sentence 2: {mse2}")

# Function to compute Word Mover's Distance (WMD) for paired sentences
def compute_wmd(sentences1, sentences2, model):
    """Compute Word Mover's Distance (WMD) for paired sentences."""
    return [1 - model.wmdistance(sent1, sent2) for sent1, sent2 in zip(sentences1, sentences2)]

# Train WMD Similarities
train_wmd = compute_wmd(train_text1, train_text2, word_vectors)

# Test WMD Similarities
test_wmd = compute_wmd(test_text1, test_text2, word_vectors)

print(f"Sample WMD Train Similarities: {train_wmd[:5]}")
print(f"Sample WMD Test Similarities: {test_wmd[:5]}")

# Compute Pearson Correlation for WMD
train_corr, _ = pearsonr(train_wmd, train_labels)
test_corr, _ = pearsonr(test_wmd, test_labels)

print(f"Pearson Correlation for Train: {train_corr:.5f}")
print(f"Pearson Correlation for Test: {test_corr:.5f}")

# Results summary
results = {
    "mse1": mse1,
    "mse2": mse2,
    "train_corr": train_corr,
    "test_corr": test_corr,
}
print("Results:", results)

# Example of loading the saved models for inference

# Load the saved Random Forest models
reg1_loaded = load(model_path1)
reg2_loaded = load(model_path2)

print("Models loaded for inference.")

# Inference on the test set using the loaded models
test_pred1_loaded = reg1_loaded.predict(test_vecs1)
test_pred2_loaded = reg2_loaded.predict(test_vecs2)

# You can now use these predictions for evaluation or further processing
print(f"Predictions for Sentence 1: {test_pred1_loaded[:5]}")
print(f"Predictions for Sentence 2: {test_pred2_loaded[:5]}")
