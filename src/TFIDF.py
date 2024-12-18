#%% md
# # 3.3.1. TF-IDF with Random Forest for Sentence Similarity
#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import joblib  # For saving models

#%%
data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic Analysis/ClinicalSTS/clinic_c.csv"
df = pd.read_csv(data_path)
df.head()

#%%
# Split into train/test sets
train_text1 = df["Sent1"][:750].tolist()
train_text2 = df["Sent2"][:750].tolist()
train_labels = df["Score"][:750].tolist()

test_text1 = df["Sent1"][750:].tolist()
test_text2 = df["Sent2"][750:].tolist()
test_labels = df["Score"][750:].tolist()

#%%
# Combine Sent1 and Sent2 for unified vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(df["Sent1"].tolist() + df["Sent2"].tolist())

# Transform train and test sets
train_vecs1 = vectorizer.transform(train_text1)
train_vecs2 = vectorizer.transform(train_text2)
test_vecs1 = vectorizer.transform(test_text1)
test_vecs2 = vectorizer.transform(test_text2)

# Store feature names for analysis
features = vectorizer.get_feature_names_out()
print(f"TF-IDF Vocabulary Size: {len(features)}")

#%%
# Train on vectorized Sent1 and Sent2
regressor1 = RandomForestRegressor(max_depth=6).fit(train_vecs1, train_labels)
regressor2 = RandomForestRegressor(max_depth=6).fit(train_vecs2, train_labels)

# Save the trained models for later inference
joblib.dump(regressor1, '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/regressor1_model.pkl')
joblib.dump(regressor2, '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/regressor2_model.pkl')
joblib.dump(vectorizer, '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/vectorizer_model.pkl')

print("Models saved successfully!")

#%%
# Predict using Random Forest models
test_pred1 = regressor1.predict(test_vecs1)
test_pred2 = regressor2.predict(test_vecs2)

# Calculate Mean Squared Error (MSE)
mse1 = mean_squared_error(test_labels, test_pred1)
mse2 = mean_squared_error(test_labels, test_pred2)

print(f"MSE for Sentence 1: {mse1}")
print(f"MSE for Sentence 2: {mse2}")

#%%
def calculate_cosine_similarity(vecs1, vecs2):
    """Calculate cosine similarity for paired vectors."""
    similarities = []
    for i in range(vecs1.shape[0]):
        similarity = cosine_similarity(vecs1[i], vecs2[i])[0][0]
        similarities.append(similarity)
    return similarities

# Compute cosine similarity for train and test sets
cosine_sim_train = calculate_cosine_similarity(train_vecs1, train_vecs2)
cosine_sim_test = calculate_cosine_similarity(test_vecs1, test_vecs2)

print(f"Sample Train Cosine Similarities: {cosine_sim_train[:5]}")
print(f"Sample Test Cosine Similarities: {cosine_sim_test[:5]}")

#%%
# Correlation for Train Data
train_corr, _ = pearsonr(cosine_sim_train, train_labels)
print(f"Pearson Correlation for Train: {train_corr:.5f}")

# Correlation for Test Data
test_corr, _ = pearsonr(cosine_sim_test, test_labels)
print(f"Pearson Correlation for Test: {test_corr:.5f}")

#%%
results = {
    "mse1": mse1,
    "mse2": mse2,
    "cosine_sim_train": cosine_sim_train,
    "cosine_sim_test": cosine_sim_test,
    "train_corr": train_corr,
    "test_corr": test_corr,
}
print("Results:", results)
