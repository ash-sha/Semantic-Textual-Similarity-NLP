import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare the dataset
DATA_DIR = os.getenv('DATA_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/data/clinic_c.csv')  # Default path if env var is not set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Load and prepare the dataset
data = pd.read_csv(DATA_DIR)  # Update with your correct path
sentences_1 = data["Sent1"].tolist()
sentences_2 = data["Sent2"].tolist()
scores = data["Score"].tolist()


# Custom Dataset class
class SimilarityDataset(Dataset):
    def __init__(self, sentences_1, sentences_2, scores, tokenizer, max_length=128):
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sentences_1[idx],
            self.sentences_2[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.scores[idx], dtype=torch.float)
        }


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                                           num_labels=1)

# Split data into training and validation sets
train_texts_1, val_texts_1, train_texts_2, val_texts_2, train_scores, val_scores = train_test_split(
    sentences_1, sentences_2, scores, test_size=0.1, random_state=42
)

# Create datasets
train_dataset = SimilarityDataset(train_texts_1, train_texts_2, train_scores, tokenizer)
val_dataset = SimilarityDataset(val_texts_1, val_texts_2, val_scores, tokenizer)


# DataLoader collate function
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

RESULT_PATH = os.getenv('RESULT_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/results')  # Default path if env var is not set
LOG_PATH = os.getenv('LOG_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/logs')  # Default path if env var is not set

# Hyperparameters for training
training_args = TrainingArguments(
    output_dir=RESULT_PATH,  # Update with your path
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # Adjust learning rate
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increase epochs for better convergence
    weight_decay=0.01,
    logging_dir=LOG_PATH,  # Update with your path
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50,
)


# Metric function for evaluation (MSE and Spearman's correlation)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(labels, predictions)

    # Compute Spearman correlation
    spearman_corr, _ = spearmanr(labels, predictions)

    return {"mse": mse, "spearman": spearman_corr}


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model after training
MODEL_DIR = os.getenv('MODEL_DIR', '/Users/aswathshakthi/PycharmProjects/MLOps/Semantic-Textual-Similarity-NLP/models/')  # Default path if env var is not set

trainer.save_model(MODEL_DIR)  # Update with your path

# Move model to the correct device
device = torch.device("cpu")  # Use 'cuda' if GPU is available
model.to(device)


# Predict similarity between two sentences
def predict_similarity(sentence1, sentence2):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            sentence1, sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()
        return score


# Example prediction
sentence_a = "Sample sentence 1"
sentence_b = "Sample sentence 2"
similarity_score = predict_similarity(sentence_a, sentence_b)
print(f"Predicted similarity score: {similarity_score}")
