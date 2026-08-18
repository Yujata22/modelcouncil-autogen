import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


df = pd.read_csv("data/churn_data.csv")

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


models = {
    "logistic_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ),
}


results = {}


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results[model_name] = {
        "accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "precision": round(
            precision_score(y_test, predictions), 4
        ),
        "recall": round(
            recall_score(y_test, predictions), 4
        ),
        "f1": round(
            f1_score(y_test, predictions), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilities), 4
        ),
    }


print("\n========== MODEL RESULTS ==========\n")

for model_name, metrics in results.items():
    print(model_name)
    print(metrics)
    print()


with open("ml/model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)


print("Saved metrics to ml/model_metrics.json")
