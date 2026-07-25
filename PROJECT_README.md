# AI-Powered Banking Support & Fraud Intelligence System

This project combines:
- NLP-based intent and sentiment classification
- A lightweight RAG-style retriever over support tickets, QA pairs, and policy documents
- A fraud-risk classifier using transaction data
- A Streamlit-based support dashboard

## Files
- `app.py`: main Streamlit application
- `src/data_utils.py`: loaders and preprocessing helpers
- `src/ml_models.py`: intent, sentiment, and fraud models
- `src/rag_engine.py`: retrieval and response generation logic
- `requirements.txt`: Python dependencies

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow
1. Enter a customer issue such as "I see a transaction of ₹10,000 I didn't make"
2. Optionally provide transaction metadata
3. The app returns:
   - predicted intent
   - detected sentiment
   - retrieved policy/FAQ evidence
   - suggested response
   - risk level and fraud probability
   - recommended support action
