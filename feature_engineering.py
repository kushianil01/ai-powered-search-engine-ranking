import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

df = pd.read_csv("clean_jobs.csv")

df = df.drop_duplicates(subset=["job_id"])

df = df.fillna("")

df["search_text"] = (
    df["title"] + " " +
    df["skills_desc"] + " " +
    df["description"]
)

corpus = [
    doc.lower().split()
    for doc in df["search_text"]
]

bm25 = BM25Okapi(corpus)

query = "python backend intern"

query_tokens = query.lower().split()

bm25_scores = bm25.get_scores(query_tokens)

features = []

for idx, row in df.iterrows():

    title = row["title"].lower()
    desc = row["description"].lower()
    skills = str(row["skills_desc"]).lower()

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

    features.append([
        row["job_id"],
        bm25_scores[idx],
        title_matches,
        desc_matches,
        skill_matches
    ])

feature_df = pd.DataFrame(
    features,
    columns=[
        "job_id",
        "bm25_score",
        "title_matches",
        "desc_matches",
        "skill_matches"
    ]
)

print(
    feature_df.sort_values(
        by="bm25_score",
        ascending=False
    ).head(10)
)
feature_df.to_csv(
    "ranking_features.csv",
    index=False
)

print("Saved ranking_features.csv")