from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.data_utils import clean_text


class IntentClassifier:
    """Train a simple text classifier for support intent."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        self.accuracy: float | None = None

    def fit(self, texts: List[str], labels: List[str]) -> None:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)

    def predict(self, text: str) -> str:
        return self.pipeline.predict([clean_text(text)])[0]

    def predict_proba(self, text: str) -> np.ndarray:
        return self.pipeline.predict_proba([clean_text(text)])[0]


class SentimentClassifier:
    """Train a simple text classifier for customer sentiment."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        self.accuracy: float | None = None

    def fit(self, texts: List[str], labels: List[str]) -> None:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)

    def predict(self, text: str) -> str:
        return self.pipeline.predict([clean_text(text)])[0]


class FraudRiskModel:
    """Train a structured fraud classifier from transaction features."""

    def __init__(self) -> None:
        numeric_features = ["amount_inr", "hour_of_day"]
        categorical_features = ["day_of_week", "is_international", "merchant_category", "transaction_type", "city", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]
        self.pipeline = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
                            (
                                "categorical",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                categorical_features,
                            ),
                        ]
                    ),
                ),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")),
            ]
        )
        self.accuracy: float | None = None

    def fit(self, transactions: pd.DataFrame) -> None:
        feature_columns = [
            "amount_inr",
            "hour_of_day",
            "day_of_week",
            "is_international",
            "merchant_category",
            "transaction_type",
            "city",
            "velocity_flag",
            "geo_anomaly_flag",
            "high_amount_flag",
        ]
        X = transactions[feature_columns]
        y = transactions["fraud_label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)

    def predict_proba(self, features: Dict[str, Any]) -> float:
        row = pd.DataFrame([features])
        probabilities = self.pipeline.predict_proba(row)[0]
        return float(probabilities[1])
