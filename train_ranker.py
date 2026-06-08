import pandas as pd
from lightgbm import LGBMRanker
import joblib

print("Loading dataset...")

df = pd.read_csv("ltr_dataset.csv")

features = [
    "bm25_score",
    "title_matches",
    "desc_matches",
    "skill_matches"
]

X = df[features]
y = df["relevance"]

group_sizes = (
    df.groupby("query")
    .size()
    .tolist()
)

print("Training LightGBM Ranker...")

model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.05,
    n_estimators=100
)

model.fit(
    X,
    y,
    group=group_sizes
)

joblib.dump(
    model,
    "ranker.pkl"
)

print("Model saved as ranker.pkl")

print("\nFeature Importance:")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(f"{feature}: {importance}")