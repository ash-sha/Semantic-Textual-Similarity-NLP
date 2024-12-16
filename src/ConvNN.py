#%% md
# # 3.3.4 CNN for Sentence Similarity
#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm

#%%
data_path = "clinic_c.csv"
word2vec_path = "/Users/aswath/PycharmProjects/mfac038/IndividualProject/PubMed-and-PMC-w2v.bin"

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
word_vec_dim = 200
oov_vec = np.random.rand(word_vec_dim)

def tokenize_and_pad(sentence_list, word_vectors, max_len=None):
    """Tokenize, vectorize, and pad sentences."""
    tokenized = [word_tokenize(sentence.lower()) for sentence in sentence_list]
    max_len = max_len or max(len(tokens) for tokens in tokenized)
    padded_vecs = [
        np.vstack([word_vectors[word] if word in word_vectors else oov_vec for word in tokens] +
                  [[0] * word_vec_dim] * (max_len - len(tokens)))
        for tokens in tokenized
    ]
    return np.array(padded_vecs)

def evaluate_model(model, data_sentences, data_labels, batch_size, word_vectors, loss_fn, device):
    """Evaluate model performance on a dataset."""
    model.eval()
    predictions = []
    losses = []
    with torch.no_grad():
        for i in range(0, len(data_sentences), batch_size):
            batch_sentences = data_sentences[i:i+batch_size]
            batch_labels = torch.tensor(data_labels[i:i+batch_size], dtype=torch.float32, device=device)

            # Prepare data
            batch_data = tokenize_and_pad(batch_sentences, word_vectors)
            batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

            # Forward pass
            preds = model(batch_data_tensor).squeeze()
            predictions.extend(preds.cpu().numpy())
            losses.append(loss_fn(preds, batch_labels).item())

    mse = mean_squared_error(data_labels, predictions)
    return mse, np.mean(losses)

#%%
class CNNRegressor(nn.Module):
    def __init__(self, embd_dim, filter_sizes, filter_nums, dropout_rate=0.5, device='cpu'):
        super(CNNRegressor, self).__init__()
        self.device = device
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embd_dim, out_channels=num_filters, kernel_size=fs, padding=fs-1)
            for fs, num_filters in zip(filter_sizes, filter_nums)
        ])
        self.fc = nn.Linear(sum(filter_nums), 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Apply convolution, activation, and pooling
        x = x.permute(0, 2, 1)  # (batch_size, embd_dim, seq_len)
        conv_out = [torch.max(self.tanh(conv(x)), dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_out, dim=1)  # Concatenate along filter dimension
        x = self.dropout(x)
        return self.fc(x)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filter_sizes = [2, 3, 4]
filter_nums = [100, 100, 100]
dropout_rate = 0.5

model = CNNRegressor(embd_dim=word_vec_dim, filter_sizes=filter_sizes, filter_nums=filter_nums, dropout_rate=dropout_rate, device=device)
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

#%%
n_epochs = 10
batch_size = 32
best_mse = float('inf')
best_model = None

for epoch in tqdm(range(n_epochs)):
    model.train()
    epoch_losses = []
    for i in range(0, len(train_sentences), batch_size):
        batch_sentences = train_sentences[i:i+batch_size]
        batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.float32, device=device)

        # Prepare data
        batch_data = tokenize_and_pad(batch_sentences, word_vectors)
        batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

        # Forward pass
        preds = model(batch_data_tensor).squeeze()
        loss = loss_fn(preds, batch_labels)
        epoch_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on dev set
    dev_mse, dev_loss = evaluate_model(model, dev_sentences, dev_labels, batch_size, word_vectors, loss_fn, device)
    print(f"Epoch {epoch+1}, Train Loss: {np.mean(epoch_losses):.4f}, Dev MSE: {dev_mse:.4f}")

    # Save best model
    if dev_mse < best_mse:
        best_mse = dev_mse
        best_model = model.state_dict()

    scheduler.step()

print(f"Best Dev MSE: {best_mse:.4f}")
model.load_state_dict(best_model)

#%%
test_mse, _ = evaluate_model(model, test_sentences, test_labels, batch_size, word_vectors, loss_fn, device)
print(f"Test MSE: {test_mse:.4f}")
