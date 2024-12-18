#%% md
# # 3.3.5 LSTM for Sentence Similarity
#%%
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import copy

#%%
data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/clinic_c.csv"
word2vec_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/PubMed-and-PMC-w2v.bin"

df = pd.read_csv(data_path)
print(df.head())

# Split into train/dev/test sets
train_text1, train_text2, train_labels = df["Sent1"][:600], df["Sent2"][:600], df["Score"][:600]
dev_text1, dev_text2, dev_labels = df["Sent1"][600:750], df["Sent2"][600:750], df["Score"][600:750]
test_text1, test_text2, test_labels = df["Sent1"][750:], df["Sent2"][750:], df["Score"][750:]

# Concatenate sentence pairs
train_sentences = [f"{s1} {s2}" for s1, s2 in zip(train_text1, train_text2)]
dev_sentences = [f"{s1} {s2}" for s1, s2 in zip(dev_text1, dev_text2)]
test_sentences = [f"{s1} {s2}" for s1, s2 in zip(test_text1, test_text2)]

#%%
word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
print(f"Word2Vec model loaded with {len(word_vectors)} words.")

#%%
embd_dim = 200
oov_vec = np.random.rand(embd_dim)

def tokenize_and_pad(sentences, word_vectors, max_len=None):
    """Tokenize, vectorize, and pad sentences."""
    tokenized = [word_tokenize(sentence.lower()) for sentence in sentences]
    max_len = max_len or max(len(tokens) for tokens in tokenized)
    padded_vecs = [
        np.vstack([word_vectors[word] if word in word_vectors else oov_vec for word in tokens] +
                  [[0] * embd_dim] * (max_len - len(tokens)))
        for tokens in tokenized
    ]
    return np.array(padded_vecs)

def evaluate_model(model, sentences, labels, batch_size, word_vectors, loss_fn, device):
    """Evaluate the model's performance."""
    model.eval()
    predictions = []
    losses = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_labels = torch.tensor(np.array(labels[i:i+batch_size]), dtype=torch.float32, device=device)
            batch_data = tokenize_and_pad(batch_sentences, word_vectors)
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

            preds = model(batch_tensor).squeeze()
            predictions.extend(preds.cpu().numpy())
            losses.append(loss_fn(preds, batch_labels).item())
    mse = mean_squared_error(labels, predictions)
    return mse, np.mean(losses)

#%%
class RNNRegressor(nn.Module):
    def __init__(self, embd_dim, hidden_dim, rnn_type="lstm", pooler_type="avg", dropout=0.5, device="cpu"):
        super(RNNRegressor, self).__init__()
        self.device = device
        assert rnn_type in ["rnn", "lstm", "bilstm", "gru"], "Invalid RNN type"
        assert pooler_type in ["max", "avg"], "Invalid pooler type"

        # RNN type
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=embd_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=embd_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout)
        elif rnn_type == "bilstm":
            self.rnn = nn.LSTM(input_size=embd_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        else:  # GRU
            self.rnn = nn.GRU(input_size=embd_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(2 * hidden_dim if rnn_type == "bilstm" else hidden_dim, 1)
        self.pooler_type = pooler_type

    def forward(self, x):
        output, _ = self.rnn(x)
        if self.pooler_type == "max":
            pooled = torch.max(output, dim=1)[0]
        else:  # Average Pooling
            pooled = torch.mean(output, dim=1)
        return self.fc(pooled)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNRegressor(embd_dim=200, hidden_dim=200, rnn_type="bilstm", pooler_type="avg", dropout=0.5, device=device)
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

#%%
n_epochs = 10
batch_size = 32
best_mse = float("inf")
best_model = None

for epoch in tqdm(range(n_epochs)):
    model.train()
    epoch_losses = []
    for i in range(0, len(train_sentences), batch_size):
        batch_sentences = train_sentences[i:i+batch_size]
        batch_labels = torch.tensor(np.array(train_labels[i:i+batch_size]), dtype=torch.float32, device=device)
        batch_data = tokenize_and_pad(batch_sentences, word_vectors)
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

        # Forward pass
        preds = model(batch_tensor).squeeze()
        loss = loss_fn(preds, batch_labels)
        epoch_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
        optimizer.step()

    # Evaluate on dev set
    dev_mse, _ = evaluate_model(model, dev_sentences, dev_labels, batch_size, word_vectors, loss_fn, device)
    print(f"Epoch {epoch+1}, Train Loss: {np.mean(epoch_losses):.4f}, Dev MSE: {dev_mse:.4f}")

    # Save best model
    if dev_mse < best_mse:
        best_mse = dev_mse
        best_model = copy.deepcopy(model.state_dict())  # Save the best model's state dict
        print(f"Best Model Updated: Dev MSE = {best_mse:.4f}")

    scheduler.step()

#%%
# Save the model weights (state_dict)
model_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/best_model_lstm.pth"
torch.save(best_model, model_path)
print(f"Best model saved to {model_path}")

#%%
# Load the saved model
model = RNNRegressor(embd_dim=200, hidden_dim=200, rnn_type="bilstm", pooler_type="avg", dropout=0.5, device=device)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Evaluate on test set
test_mse, _ = evaluate_model(model, test_sentences, test_labels, batch_size, word_vectors, loss_fn, device)
print(f"Test MSE: {test_mse:.4f}")
