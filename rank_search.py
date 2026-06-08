import pandas as pd
import numpy as np
import joblib
from rank_bm25 import BM25Okapi

print("Loading jobs...")

# Load data
df = pd.read_csv("clean_jobs.csv")

df = df.drop_duplicates(subset=["job_id"])
df = df.fillna("")

# Load trained ranker
ranker = joblib.load("ranker.pkl")

# Create searchable text
df["search_text"] = (
    df["title"] + " " +
    df["skills_desc"] + " " +
    df["description"]
)

# Build BM25 index
corpus = [
    doc.lower().split()
    for doc in df["search_text"]
]

bm25 = BM25Okapi(corpus)

print("Rank Search Ready!")

while True:

    query = input("\nSearch jobs: ")

    if query.lower() == "exit":
        break

    query_tokens = query.lower().split()

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query_tokens)

    # Get top 100 candidates
    candidate_indices = np.argsort(bm25_scores)[::-1][:100]

    feature_rows = []

    for idx in candidate_indices:

        title = str(df.iloc[idx]["title"]).lower()
        desc = str(df.iloc[idx]["description"]).lower()
        skills = str(df.iloc[idx]["skills_desc"]).lower()

        title_matches = sum(
            word in title
            for word in query_tokens
        )

        desc_matches = sum(
            word in desc
            for word in query_tokens
        )

        skill_matches = sum(
            word in skills
            for word in query_tokens
        )

        feature_rows.append([
            bm25_scores[idx],
            title_matches,
            desc_matches,
            skill_matches
        ])

    X = pd.DataFrame(
        feature_rows,
        columns=[
            "bm25_score",
            "title_matches",
            "desc_matches",
            "skill_matches"
        ]
    )

    # Predict ranking scores
    ranking_scores = ranker.predict(X)

    # Sort by ranking score
    reranked = sorted(
        zip(candidate_indices, ranking_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Results:\n")

    for rank, (idx, score) in enumerate(reranked[:10], start=1):

        print(f"{rank}. {df.iloc[idx]['title']}")
        print(f"   Company: {df.iloc[idx]['company_name']}")
        print(f"   Location: {df.iloc[idx]['location']}")
        print(f"   Rank Score: {score:.4f}")
        print()