from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_utils import clean_text


class SimpleRAG:
    """A lightweight retriever that mimics a simple vector DB using TF-IDF."""

    def __init__(self, documents: List[dict]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.corpus_text = [clean_text(item["text"]) for item in documents]
        self.matrix = self.vectorizer.fit_transform(self.corpus_text)

    def retrieve(self, query: str, top_k: int = 4) -> List[dict]:
        query_vector = self.vectorizer.transform([clean_text(query)])
        similarity_scores = cosine_similarity(query_vector, self.matrix).ravel()
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        return [self.documents[index] for index in top_indices if similarity_scores[index] > 0.0]


def build_response(query: str, intent: str, sentiment: str, risk_level: str, retrieved_docs: List[dict]) -> str:
    """Compose a context-aware support response from the retrieved documents."""
    top_doc = retrieved_docs[0] if retrieved_docs else None
    if top_doc and "answer" in top_doc:
        base_template = top_doc["answer"]
    elif top_doc:
        base_template = top_doc["text"]
    else:
        base_template = "We do not have enough context yet. Please contact the helpdesk for a manual review."

    action_note = ""
    if intent == "Fraud":
        action_note = "Please secure the account immediately and escalate to the fraud team if the transaction appears unauthorized."
    elif intent == "Loan":
        action_note = "Please share the application number and any supporting documents if the issue is about disbursement or rejection."
    elif intent == "KYC":
        action_note = "Please ensure your documents are available and follow the KYC update procedure if your account is restricted."
    else:
        action_note = "Please provide any additional details so we can route the case correctly."

    return (
        f"Based on the retrieved banking guidance, here is a suggested response:\n"
        f"{base_template}\n\n"
        f"Customer sentiment: {sentiment}. Risk level: {risk_level}. \n"
        f"{action_note}"
    )
