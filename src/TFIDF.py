#%%
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import joblib  # For saving models
from spacy.compat import pickle

#%%
DATA_DIR = os.getenv('DATA_DIR', "/Users/aswathshakthi/PycharmProjects/MLOps/3.Semantic-Textual-Similarity-NLP/data/clinic_c.csv")  # Default path if env var is not set

df = pd.read_csv(DATA_DIR)
df.head()

#%%
# Split into train/test sets
train_num = int(df.shape[0]*0.9)

train = df[["Sent1","Sent2"]][:train_num]
train_labels = df["Score"][:train_num]

dev= df[["Sent1","Sent2"]][train_num:]
dev_labels = df["Score"][train_num:]

#%%
# Combine Sent1 and Sent2 for unified vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(train["Sent1"].tolist()+train["Sent2"].tolist())

# Save the vectorizer model
VEC_DIR = os.getenv('VEC_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/3.Semantic-Textual-Similarity-NLP/models/vectorizer_model.pkl')  # Define path to save the vectorizer
joblib.dump(vectorizer, VEC_DIR)

print(f"Vectorizer model saved to {VEC_DIR}")

#%%
# Transform train and test sets
train_vecs1 = vectorizer.transform(train.Sent1.tolist())
train_vecs2 = vectorizer.transform(train.Sent2.tolist())

dev_vecs1 = vectorizer.transform(dev.Sent1.tolist())
dev_vecs2 = vectorizer.transform(dev.Sent2.tolist())


# Store feature names for analysis
features = vectorizer.get_feature_names_out()
print(f"TF-IDF Vocabulary Size: {len(features)}")
#%%

train_features = np.hstack([train_vecs1.toarray(), train_vecs2.toarray()])
test_features = np.hstack([dev_vecs1.toarray(), dev_vecs2.toarray()])


def calculate_cosine_similarity(vecs1, vecs2):
    return np.array([cosine_similarity(vecs1[i], vecs2[i])[0][0] for i in range(vecs1.shape[0])])

train_cosine_sim = calculate_cosine_similarity(train_vecs1, train_vecs2)
dev_cosine_sim = calculate_cosine_similarity(dev_vecs1, dev_vecs2)

# Append cosine similarity as a feature
train_features = np.hstack([train_vecs1.toarray(), train_vecs2.toarray(), train_cosine_sim.reshape(-1, 1)])
dev_features = np.hstack([dev_vecs1.toarray(), dev_vecs2.toarray(), dev_cosine_sim.reshape(-1, 1)])


#%%

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 1000),
        max_depth=trial.suggest_int('max_depth', 5, 100),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    )
    scores = cross_val_score(rf, train_features, train_labels, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize',study_name="Similarity", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=50,n_jobs=-1)

print("Best Parameters:", study.best_params)
best_model = RandomForestRegressor(**study.best_params)

best_model.fit(train_features, train_labels)


# Save the trained models for later inference
MODEL_DIR = os.getenv('MODEL_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/3.Semantic-Textual-Similarity-NLP/models/tfidf_rf_model.pkl')
joblib.dump(best_model, MODEL_DIR)


print("Models saved successfully!")

#%%
# Predict using Random Forest models
model = joblib.load(MODEL_DIR)
dev_pred = model.predict(dev_features)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(dev_labels, dev_pred)

mse
#%%
# Correlation for Train Data
train_corr, _ = pearsonr(train_cosine_sim, train_labels)
print(f"Pearson Correlation for Train: {train_corr:.5f}")

# Correlation for Test Data
test_corr, _ = pearsonr(dev_cosine_sim, dev_labels)
print(f"Pearson Correlation for Test: {test_corr:.5f}")

#%%
results = {
    "mse": mse,
    "train_corr": train_corr,
    "test_corr": test_corr,
}
print("Results:", results)

#%%
print(vectorizer.vocabulary_)


#%%
# Inference
vectorizer = joblib.load(VEC_DIR)
rf_model = joblib.load(MODEL_DIR)

def calculate_cosine_similarity(vecs1, vecs2):
    return np.array([cosine_similarity(vecs1[i], vecs2[i])[0][0] for i in range(vecs1.shape[0])])

def predict_sim(sentence1: str, sentence2: str):
    # Vectorize the input sentences using the loaded vectorizer
    vec1 = vectorizer.transform([sentence1])
    vec2 = vectorizer.transform([sentence2])

    # Combine the sentence vectors (TF-IDF) for prediction
    in_sim = calculate_cosine_similarity(vec1, vec2)
    features = np.hstack([vec1.toarray(), vec2.toarray(), in_sim.reshape(-1, 1)])

    # Predict the similarity score using the Random Forest model
    predicted_score = rf_model.predict(features)[0]

    return predicted_score

predicted_score = predict_sim("I love India", "I love India")
print(f"Predicted Similarity Score: {predicted_score}")