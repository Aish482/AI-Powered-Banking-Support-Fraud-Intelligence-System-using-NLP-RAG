from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src.data_utils import (
    build_retrieval_corpus,
    extract_amount,
    load_policy_documents,
    load_qa_pairs,
    load_ticket_data,
    load_transaction_data,
    normalize_category,
)
from src.ml_models import FraudRiskModel, IntentClassifier, SentimentClassifier
from src.rag_engine import SimpleRAG, build_response


@st.cache_resource
def load_models() -> Dict[str, Any]:
    tickets = load_ticket_data()
    transactions = load_transaction_data()
    qa_pairs = load_qa_pairs()
    policy_documents = load_policy_documents()
    retrieval_corpus = build_retrieval_corpus(tickets, qa_pairs, policy_documents)

    intent_model = IntentClassifier()
    intent_model.fit(
        tickets["query_text"].tolist(),
        tickets["category_clean"].tolist(),
    )

    sentiment_model = SentimentClassifier()
    sentiment_model.fit(
        tickets["query_text"].tolist(),
        tickets["sentiment"].tolist(),
    )

    fraud_model = FraudRiskModel()
    fraud_model.fit(transactions)
    rag = SimpleRAG(retrieval_corpus)

    return {
        "tickets": tickets,
        "transactions": transactions,
        "qa_pairs": qa_pairs,
        "policy_documents": policy_documents,
        "retrieval_corpus": retrieval_corpus,
        "intent_model": intent_model,
        "sentiment_model": sentiment_model,
        "fraud_model": fraud_model,
        "rag": rag,
    }


def estimate_risk(query: str, intent: str, sentiment: str, amount: float | None) -> str:
    text = query.lower()
    if intent == "Fraud":
        if amount is not None and amount > 50000:
            return "High"
        if any(word in text for word in ["otp", "upi", "unauthorized", "phishing", "sim swap", "without my consent"]):
            return "High"
        if any(word in text for word in ["duplicate", "dispute", "subscription"]):
            return "Medium"
        return "High"
    if intent == "Loan":
        if any(word in text for word in ["rejected", "approved", "delay", "disbursement"]):
            return "Medium"
        return "Low"
    if intent == "KYC":
        if any(word in text for word in ["restricted", "freeze", "urgent"]):
            return "Medium"
        return "Low"
    if sentiment in {"Urgent", "Frustrated", "Angry", "Anxious"}:
        return "Medium"
    return "Low"


def build_action(intent: str, risk_level: str, amount: float | None) -> str:
    if intent == "Fraud":
        if risk_level == "High":
            return "Block card/account, raise a dispute, and escalate to the fraud team."
        return "Review the transaction and offer provisional credit if the dispute is valid."
    if intent == "Loan":
        return "Check the loan application status and request any missing documents."
    if intent == "KYC":
        return "Request the missing KYC documents and advise branch follow-up if the account is restricted."
    return "Route the query to the relevant support queue for manual review."


def analyze_query(query: str, amount: float | None, merchant: str, merchant_category: str, transaction_type: str, city: str, hour: int, day: str, is_international: bool, velocity_flag: bool, geo_flag: bool, high_amount_flag: bool) -> Dict[str, Any]:
    models = load_models()
    intent_model = models["intent_model"]
    sentiment_model = models["sentiment_model"]
    fraud_model = models["fraud_model"]
    rag = models["rag"]

    intent = intent_model.predict(query)
    sentiment = sentiment_model.predict(query)
    if amount is None:
        amount = extract_amount(query)
    risk_level = estimate_risk(query, intent, sentiment, amount)
    action = build_action(intent, risk_level, amount)
    retrieved_docs = rag.retrieve(query, top_k=4)
    response = build_response(query, intent, sentiment, risk_level, retrieved_docs)

    features = {
        "amount_inr": float(amount or 0),
        "hour_of_day": hour,
        "day_of_week": day or "Unknown",
        "is_international": bool(is_international),
        "merchant_category": merchant_category or "Unknown",
        "transaction_type": transaction_type or "UPI",
        "city": city or "Unknown",
        "velocity_flag": bool(velocity_flag),
        "geo_anomaly_flag": bool(geo_flag),
        "high_amount_flag": bool(high_amount_flag),
    }
    fraud_probability = fraud_model.predict_proba(features)
    fraud_label = "High risk" if fraud_probability > 0.6 else "Medium risk" if fraud_probability > 0.35 else "Low risk"

    return {
        "intent": intent,
        "sentiment": sentiment,
        "risk_level": risk_level,
        "action": action,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "fraud_probability": fraud_probability,
        "fraud_label": fraud_label,
        "intent_accuracy": intent_model.accuracy,
        "sentiment_accuracy": sentiment_model.accuracy,
        "fraud_accuracy": fraud_model.accuracy,
    }


def main() -> None:
    st.set_page_config(page_title="Banking Support & Fraud Intelligence", page_icon="🏦", layout="wide")
    st.title("AI-Powered Banking Support & Fraud Intelligence")
    st.caption("NLP + RAG prototype using support tickets, policy documents, QA pairs, and transaction data")

    with st.sidebar:
        st.header("Customer query")
        query_text = st.text_area(
            "Customer query",
            value="I see a transaction of ₹10,000 I didn't make",
            help="Enter a customer issue or fraud-related complaint",
        )
        amount = st.number_input("Amount (₹)", min_value=0.0, value=10000.0, step=100.0)
        merchant = st.text_input("Merchant name", value="UnknownMerchant")
        merchant_category = st.text_input("Merchant category", value="Unknown")
        transaction_type = st.selectbox("Transaction type", ["UPI", "Debit Card", "Credit Card", "Net Banking", "ATM", "NEFT"])
        city = st.text_input("City", value="Delhi")
        hour = st.slider("Hour of day", 0, 23, 10)
        day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        is_international = st.checkbox("International transaction")
        velocity_flag = st.checkbox("Velocity anomaly")
        geo_flag = st.checkbox("Geo anomaly")
        high_amount_flag = st.checkbox("High amount")

        analyze_button = st.button("Analyze query", use_container_width=True)

    if analyze_button:
        with st.spinner("Training and retrieving context..."):
            result = analyze_query(
                query=query_text,
                amount=amount,
                merchant=merchant,
                merchant_category=merchant_category,
                transaction_type=transaction_type,
                city=city,
                hour=hour,
                day=day,
                is_international=is_international,
                velocity_flag=velocity_flag,
                geo_flag=geo_flag,
                high_amount_flag=high_amount_flag,
            )

        st.subheader("Live result")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Intent", result["intent"])
        col2.metric("Sentiment", result["sentiment"])
        col3.metric("Risk", result["risk_level"])
        col4.metric("Fraud label", result["fraud_label"])

        st.metric("Fraud probability", f"{result['fraud_probability'] * 100:.1f}%")

        st.write("### Suggested response")
        st.info(result["response"])

        st.write("### Suggested action")
        st.success(result["action"])

        st.write("### Retrieved evidence")
        for index, doc in enumerate(result["retrieved_docs"], start=1):
            with st.expander(f"{index}. {doc['source'].upper()} — {doc['title']}"):
                st.write(doc["text"])

        st.write("### Model summary")
        st.caption(f"Intent classifier accuracy: {result['intent_accuracy']:.2f}")
        st.caption(f"Sentiment classifier accuracy: {result['sentiment_accuracy']:.2f}")
        st.caption(f"Fraud model accuracy: {result['fraud_accuracy']:.2f}")
    else:
        st.info("Enter a customer query and click Analyze query to see the support workflow in action.")


if __name__ == "__main__":
    main()
