from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1]

TICKETS_PATH = DATA_DIR / "01_support_tickets.csv"
TRANSACTIONS_PATH = DATA_DIR / "02_transactions.csv"
QA_PATH = DATA_DIR / "04_qa_pairs.json"
POLICY_PATHS = {
    "Fraud": DATA_DIR / "fraud_handling_policy.txt",
    "KYC": DATA_DIR / "kyc_policy.txt",
    "Loan": DATA_DIR / "loan_processing_policy.txt",
    "Refund": DATA_DIR / "refund_dispute_policy.txt",
}


def load_ticket_data() -> pd.DataFrame:
    """Load the synthetic support tickets dataset."""
    tickets = pd.read_csv(TICKETS_PATH)
    tickets["category_clean"] = tickets["category"].apply(normalize_category)
    return tickets


def load_transaction_data() -> pd.DataFrame:
    """Load the synthetic transaction dataset."""
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    transactions["day_of_week"] = transactions["day_of_week"].fillna("Unknown")
    transactions["is_international"] = transactions["is_international"].replace({"Yes": True, "No": False})
    transactions["velocity_flag"] = transactions["velocity_flag"].replace({"Yes": True, "No": False})
    transactions["geo_anomaly_flag"] = transactions["geo_anomaly_flag"].replace({"Yes": True, "No": False})
    transactions["high_amount_flag"] = transactions["high_amount_flag"].replace({"Yes": True, "No": False})
    return transactions


def load_qa_pairs() -> List[dict]:
    """Load the QA evaluation pairs."""
    with QA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_policy_documents() -> Dict[str, str]:
    """Load policy documents used for retrieval."""
    return {name: path.read_text(encoding="utf-8") for name, path in POLICY_PATHS.items()}


def normalize_category(category: str) -> str:
    """Normalize the ticket category into 4 high-level classes."""
    value = str(category).lower()
    if "fraud" in value:
        return "Fraud"
    if "loan" in value:
        return "Loan"
    if "kyc" in value:
        return "KYC"
    if "account" in value:
        return "Account Access"
    return "General"


def extract_amount(query: str) -> float | None:
    """Extract an INR amount from a customer query when present."""
    if not query:
        return None
    matches = re.findall(r"₹?\s?([0-9,]+(?:\.\d{1,2})?)", query)
    if not matches:
        return None
    amount_text = matches[-1].replace(",", "")
    try:
        return float(amount_text)
    except ValueError:
        return None


def clean_text(text: str) -> str:
    """Basic text cleaning for NLP models."""
    if not text:
        return ""
    text = text.replace("₹", "rupees ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def split_policy_into_chunks(policy_text: str, chunk_size: int = 350) -> List[str]:
    """Split policy text into smaller retrieval chunks."""
    sentences = re.split(r"(?<=[.])\s+", policy_text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def build_retrieval_corpus(tickets: pd.DataFrame, qa_pairs: List[dict], policy_documents: Dict[str, str]) -> List[dict]:
    """Create a retrieval corpus from tickets, QA pairs, and policy docs."""
    docs: List[dict] = []

    for _, row in tickets.iterrows():
        docs.append(
            {
                "text": f"{row['query_text']} | Resolution: {row['resolution_text']}",
                "category": normalize_category(row["category"]),
                "source": "ticket",
                "title": row["ticket_id"],
            }
        )

    for qa in qa_pairs:
        docs.append(
            {
                "text": f"{qa['question']} | {qa['answer']} | Policy: {qa['policy_ref']}",
                "category": qa["category"],
                "source": "qa",
                "title": qa["id"],
                "answer": qa["answer"],
            }
        )

    for name, policy_text in policy_documents.items():
        chunks = split_policy_into_chunks(policy_text)
        for chunk in chunks:
            docs.append(
                {
                    "text": chunk,
                    "category": name,
                    "source": "policy",
                    "title": f"{name} policy excerpt",
                }
            )

    return docs
