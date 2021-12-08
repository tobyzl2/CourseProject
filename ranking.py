import json

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# citation: https://stackoverflow.com/questions/55677314/using-sklearn-how-do-i-calculate-the-tf-idf-cosine-similarity-between-documents
from transformers import BertTokenizer, BertModel

from model import ClassificationModel

mapping = [
    "artificial+intelligence",
    "hardware+architecture",
    "computer+security",
    "databases",
    "formal+languages+and+automata+theory",
    "operating+systems",
    "computer+networking",
    "numerical+analysis"
]

def run_ranking(query_title, query_summary, k=10):
    # load dataset
    with open("./dataset.json") as f:
        dataset = json.load(f)

    # load bert
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")

    # load classification model
    state_dict = torch.load("./model.pt")
    model = ClassificationModel()
    model.load_state_dict(state_dict)
    model.eval()

    # get prediction
    input = query_title + ". " + query_summary
    tokens = tokenizer(input, add_special_tokens=False, padding=True, return_tensors="pt")
    embeddings = bert(tokens["input_ids"], attention_mask=tokens["attention_mask"]).pooler_output
    logits = model(embeddings)
    pred = torch.argmax(logits).item()

    # fit tfidf
    docs = [v["title"] + ". " + v["summary"] for v in dataset[mapping[pred]]]
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(docs)

    def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
        """
        vectorizer: TfIdfVectorizer model
        docs_tfidf: tfidf vectors for all docs
        query: query doc

        return: cosine similarity between query and all docs
        """
        query_tfidf = vectorizer.transform([query])
        cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
        return cosineSimilarities

    # get rankings
    rank_idxs = np.argsort(get_tf_idf_query_similarity(vectorizer, docs_tfidf, input))
    rankings = []
    for idx in rank_idxs:
        rankings.append(docs[idx])
    return rankings[::-1][:k]


if __name__ == "__main__":
    query_title = "Artificial intelligence in medicine"
    query_summary = "Artificial intelligence is a branch of computer science capable of analysing complex medical data. Their potential to exploit meaningful relationship with in a data set can be used in the diagnosis, treatment and predicting outcome in many clinical scenarios."
    print(run_ranking(query_title, query_summary))