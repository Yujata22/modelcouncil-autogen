import pandas as pd
from sklearn.datasets import make_classification


def generate_data():
    X, y = make_classification(
        n_samples=3000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.72, 0.28],
        class_sep=1.0,
        random_state=42
    )

    feature_names = [
        "monthly_spend",
        "tenure_months",
        "support_tickets",
        "usage_frequency",
        "payment_delay",
        "contract_length",
        "feature_usage",
        "login_frequency",
        "avg_session_time",
        "discount_usage",
        "service_calls",
        "engagement_score",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["churn"] = y

    df.to_csv("data/churn_data.csv", index=False)

    print("Dataset created successfully.")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nTarget distribution:")
    print(df["churn"].value_counts(normalize=True))


if __name__ == "__main__":
    generate_data()
