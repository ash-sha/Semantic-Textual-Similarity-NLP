import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import sent2vec
import pickle
import os
import matplotlib.pyplot as plt


# Helper Functions
def embed_sentences(sentences, model):
    """
    Generate embeddings for a list of sentences using BioSentVec.
    Ensure the output is a 2D array (samples × embedding dimensions).
    """
    return np.vstack([model.embed_sentence(sentence) for sentence in sentences])



def calculate_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity for two sets of embeddings.
    """
    return [1 - cosine(emb1[i], emb2[i]) for i in range(len(emb1))]


def save_results(results, output_path="results_biosentvec_rf.json"):
    """
    Save results to a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_path}")


def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load a trained model from a file.
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def plot_cosine_similarity(train_cosine_sim, test_cosine_sim):
    """
    Plot cosine similarity distributions for train and test sets.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(train_cosine_sim, bins=20, alpha=0.7, label="Train")
    plt.hist(test_cosine_sim, bins=20, alpha=0.7, label="Test")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_biosentvec_rf_workflow(data_path, biosentvec_path, output_dir="/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models"):
    """
    Run the BioSentVec + Random Forest workflow.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    train_text1 = df["Sent1"][:750].tolist()
    train_text2 = df["Sent2"][:750].tolist()
    train_labels = df["Score"][:750].tolist()
    test_text1 = df["Sent1"][750:].tolist()
    test_text2 = df["Sent2"][750:].tolist()
    test_labels = df["Score"][750:].tolist()

    # Load BioSentVec model
    model = sent2vec.Sent2vecModel()
    model.load_model(biosentvec_path)
    print("BioSentVec model successfully loaded!")

    # Generate embeddings (ensure 2D shape)
    train_emb1 = embed_sentences(train_text1, model)
    train_emb2 = embed_sentences(train_text2, model)
    test_emb1 = embed_sentences(test_text1, model)
    test_emb2 = embed_sentences(test_text2, model)

    print(f"Train Embedding Shapes: {train_emb1.shape}, {train_emb2.shape}")
    print(f"Test Embedding Shapes: {test_emb1.shape}, {test_emb2.shape}")

    # Train Random Forest models
    reg1 = RandomForestRegressor(max_depth=6).fit(train_emb1, train_labels)
    reg2 = RandomForestRegressor(max_depth=6).fit(train_emb2, train_labels)

    # Save trained models
    save_model(reg1, os.path.join(output_dir, "rf_model_sent1.pkl"))
    save_model(reg2, os.path.join(output_dir, "rf_model_sent2.pkl"))

    # Evaluate models
    test_pred1 = reg1.predict(test_emb1)
    test_pred2 = reg2.predict(test_emb2)
    mse1 = mean_squared_error(test_labels, test_pred1)
    mse2 = mean_squared_error(test_labels, test_pred2)
    print(f"MSE for Sentence 1: {mse1}")
    print(f"MSE for Sentence 2: {mse2}")

    # Compute cosine similarity
    train_cosine_sim = calculate_cosine_similarity(train_emb1, train_emb2)
    test_cosine_sim = calculate_cosine_similarity(test_emb1, test_emb2)

    # Pearson correlation
    train_corr, _ = pearsonr(train_cosine_sim, train_labels)
    test_corr, _ = pearsonr(test_cosine_sim, test_labels)
    print(f"Pearson Correlation for Train: {train_corr:.5f}")
    print(f"Pearson Correlation for Test: {test_corr:.5f}")

    # Save results
    results = {
        "mse1": mse1,
        "mse2": mse2,
        "train_corr": train_corr,
        "test_corr": test_corr,
    }
    save_results(results, output_path=os.path.join(output_dir, "results_biosentvec_rf.json"))

    # Plot cosine similarity distributions
    plot_cosine_similarity(train_cosine_sim, test_cosine_sim)

    return results



# Run Workflow
if __name__ == "__main__":
    data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/clinic_c.csv"  # Path to your dataset
    biosentvec_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"  # Path to BioSentVec model
    results = run_biosentvec_rf_workflow(data_path, biosentvec_path)
    print("Workflow completed. Results:", results)
