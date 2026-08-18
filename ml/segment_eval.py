import json
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split
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


model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ---------------------------------------
# Build evaluation dataframe
# ---------------------------------------

eval_df = X_test.copy()

eval_df["actual"] = y_test.values
eval_df["prediction"] = predictions


# ---------------------------------------
# Create business-like segments
# ---------------------------------------

eval_df["customer_value_segment"] = pd.qcut(
    eval_df["monthly_spend"],
    q=3,
    labels=[
        "low_value",
        "medium_value",
        "high_value"
    ]
)


eval_df["engagement_segment"] = pd.qcut(
    eval_df["engagement_score"],
    q=3,
    labels=[
        "low_engagement",
        "medium_engagement",
        "high_engagement"
    ]
)


# ---------------------------------------
# Segment evaluation function
# ---------------------------------------

def evaluate_segment(df, segment_column):

    results = {}

    for segment_name, segment_df in df.groupby(
        segment_column,
        observed=True
    ):

        y_true = segment_df["actual"]
        y_pred = segment_df["prediction"]

        results[str(segment_name)] = {

            "sample_size": len(segment_df),

            "precision": round(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "recall": round(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),

            "f1": round(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),
                4
            ),
        }

    return results


segment_results = {

    "customer_value_segment":
        evaluate_segment(
            eval_df,
            "customer_value_segment"
        ),

    "engagement_segment":
        evaluate_segment(
            eval_df,
            "engagement_segment"
        )
}


print(
    "\n========== SEGMENT PERFORMANCE ==========\n"
)

print(
    json.dumps(
        segment_results,
        indent=4
    )
)


with open(
    "ml/segment_metrics.json",
    "w"
) as f:

    json.dump(
        segment_results,
        f,
        indent=4
    )


print(
    "\nSaved segment metrics to "
    "ml/segment_metrics.json"
)
