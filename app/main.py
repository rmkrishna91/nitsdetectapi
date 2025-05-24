from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import string
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import en_core_web_sm

# Load once at startup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

nlp = en_core_web_sm.load()
sentence_model = SentenceTransformer("app/embedding/all-mpnet-base-v2").to(device)

# Define model architecture
class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_size = 795
model = ClassificationModel(input_size).to(device)
model.load_state_dict(torch.load("app/classification_model1.pth", map_location=device))
model.eval()

# Load scaler
scaler = StandardScaler()
scaler.mean_ = np.load("app/scaler_mean.npy")
scaler.scale_ = np.load("app/scaler_scale.npy")

FEATURE_COLUMNS = [
    "avg line length", "vocab", "word density", "stopwords_count",
    "word count", "active", "passive", "punctuation_count",
    "linking_words_count", "NOUN", "VERB", "PUNCT", "DET", "PRON", "PROPN",
    "ADJ", "AUX", "ADV", "PART", "SCONJ", "NUM", "X", "INTJ", "ADP", "SYM",
    "SPACE", "CCONJ"
]

app = FastAPI()

class TextInput(BaseModel):
    text: str

# ----------- Text Feature Extraction Functions -----------
def calculate(text):
    def tokenize_text(t): return word_tokenize(t) if isinstance(t, str) else []
    def sent_wala(t): return sent_tokenize(t) if isinstance(t, str) else []
    def count_stop_words(words): return sum(1 for token in nlp(" ".join(words)) if token.is_stop)
    def is_passive(doc): return any(tok.dep_ == "nsubjpass" for tok in doc)
    def count_active(sents): return sum(not is_passive(nlp(s)) for s in sents)
    def count_nouns_verbs(words):
        tags = pos_tag(words)
        counts = Counter(tag for _, tag in tags)
        return {
            'NOUN': counts.get('NN', 0) + counts.get('NNS', 0) + counts.get('NNP', 0) + counts.get('NNPS', 0),
            'VERB': sum(counts.get(tag, 0) for tag in ['VB','VBD','VBG','VBN','VBP','VBZ']),
            'PUNCT': sum(counts.get(p, 0) for p in ['.', ',', ':', '(', ')', '"', "''", '``', '!', '?', ';', '-']),
            'DET': sum(counts.get(tag, 0) for tag in ['DT','PDT','WDT']),
            'PRON': counts.get('PRP', 0) + counts.get('PRP$', 0),
            'PROPN': counts.get('NNP', 0) + counts.get('NNPS', 0),
            'ADJ': sum(counts.get(tag, 0) for tag in ['JJ','JJR','JJS']),
            'AUX': counts.get('MD', 0),
            'ADV': sum(counts.get(tag, 0) for tag in ['RB','RBR','RBS']),
            'PART': counts.get('RP', 0),
            'SCONJ': counts.get('IN', 0),
            'NUM': counts.get('CD', 0),
            'X': counts.get('FW', 0),
            'INTJ': counts.get('UH', 0),
            'ADP': counts.get('IN', 0),
            'SYM': counts.get('SYM', 0),
            'SPACE': counts.get('SP', 0),
            'CCONJ': counts.get('CC', 0)
        }

    def count_punctuation(text): return sum(1 for ch in text if ch in string.punctuation)
    def count_linking(text): return sum(1 for w in word_tokenize(text.lower()) if w in {'to', 'the', 'and', 'of', 'in', 'on', 'for', 'with', 'at', 'a', 'an'})

    words = tokenize_text(text)
    sentences = sent_wala(text)
    vocab = len(set(words))
    line_count = len(sentences)
    word_count = len(words)
    avg_line_len = word_count / line_count if line_count else 0
    stop_count = count_stop_words(words)
    active = count_active(sentences)
    passive = line_count - active
    pos_counts = count_nouns_verbs(words)
    punct_count = count_punctuation(text)
    link_count = count_linking(text)
    word_density = (vocab * 100) / (avg_line_len * line_count) if avg_line_len * line_count else 0

    data = {
        "avg line length": avg_line_len,
        "vocab": vocab,
        "word density": word_density,
        "stopwords_count": stop_count,
        "word count": word_count,
        "active": active,
        "passive": passive,
        "punctuation_count": punct_count,
        "linking_words_count": link_count,
    }
    data.update(pos_counts)
    return pd.DataFrame([data]).reindex(columns=FEATURE_COLUMNS, fill_value=0)

# ----------- Prediction Function -----------
def predict(text, df):
    df = df[FEATURE_COLUMNS]
    embedding = sentence_model.encode([text], convert_to_tensor=True).to(device)
    num_feats = torch.tensor(scaler.transform(df.values), dtype=torch.float32).to(device)
    x_combined = torch.cat((embedding, num_feats), dim=1)
    with torch.no_grad():
        output = model(x_combined)
        prob = torch.sigmoid(output).cpu().numpy().flatten()
    pred = (prob > 0.5).astype(int)
    return int(pred[0]), float(prob[0])

# ----------- API Endpoint -----------
@app.post("/predict")
async def predict_route(input: TextInput):
    df = calculate(input.text)
    prediction, probability = predict(input.text, df)
    return {
        "prediction": prediction,
        "probability": probability
    }
