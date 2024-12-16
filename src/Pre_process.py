#%% md
# # CS5821 - INDIVIDUAL PROJECT
# ### Measuring Sentence Similarity in Biomedical Domain using Deep Learning Models
# #### 3. Experiment
# #### 3.2 Preprocessing
# 
#%%
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from string import punctuation

#%%
data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic Analysis/ClinicalSTS"
df0 = pd.read_csv(
    os.path.join(data_path, "clinicalSTS.train.txt"),
    sep='\t',
    header=None,
    names=["Sent1", "Sent2", "Score"]
)
df0

#%%
data_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Semantic Analysis/ClinicalSTS"
df1 = pd.read_csv(
    os.path.join(data_path, "clinicalSTS.test.txt"),
    sep='\t',
    header=None,
    names=["Sent1", "Sent2", "Score"]
)
df1["Score"] = pd.read_csv("/Users/aswathshakthi/PycharmProjects/MLOps/Semantic Analysis/ClinicalSTS/clinicalSTS.test.gs.sim.txt",header=None,names=["Score"])
df1
#%%
df = pd.concat([df0, df1],ignore_index=True)
df
#%%
def preprocess_text(series, custom_stopwords=[]):
    """
    Perform the full preprocessing pipeline on a Pandas Series:
    - Lowercasing
    - Removing numbers and punctuation
    - Tokenizing
    - Lemmatizing
    - Removing stopwords
    - Detokenizing
    """
    # Initialize tools
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')).union(custom_stopwords)

    def clean_and_process(sentence):
        # Lowercase and clean
        sentence = sentence.lower()
        sentence = ''.join(c for c in sentence if not c.isdigit())
        sentence = ''.join(c for c in sentence if c not in punctuation)

        # Tokenize
        tokens = tokenizer.tokenize(sentence)

        # Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Detokenize
        return detokenizer.detokenize(tokens)

    return series.apply(clean_and_process)

#%%
custom_stopwords_path = os.path.join(data_path, "stopwords")
custom_stopwords = (
    [line.strip() for line in open(custom_stopwords_path, 'r')]
    if os.path.exists(custom_stopwords_path)
    else []
)

#%%
df["Sent1_Processed"] = preprocess_text(df["Sent1"], custom_stopwords)
df["Sent2_Processed"] = preprocess_text(df["Sent2"], custom_stopwords)

#%%
sent1_tokens = sum(len(sentence.split()) for sentence in df["Sent1_Processed"])
sent2_tokens = sum(len(sentence.split()) for sentence in df["Sent2_Processed"])
print(f"Sentence 1 has {sent1_tokens} tokens.")
print(f"Sentence 2 has {sent2_tokens} tokens.")
print(f"Total tokens: {sent1_tokens + sent2_tokens}")

#%%
output_train_path = os.path.join(data_path, "train.csv")
output_test_path = os.path.join(data_path, "test.csv")
output_combined_path = os.path.join(data_path, "clinic_c.csv")

# Split into train and test
train_df = df.iloc[:750]
test_df = df.iloc[750:]

train_df[["Sent1", "Sent2", "Score"]].to_csv(output_train_path, index=False)
test_df[["Sent1", "Sent2", "Score"]].to_csv(output_test_path, index=False)
df[["Sent1", "Sent2", "Score"]].to_csv(output_combined_path, index=False)

print("Data exported successfully.")

#%%
