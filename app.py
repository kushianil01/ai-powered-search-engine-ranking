from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import joblib

from rank_bm25 import BM25Okapi

app = FastAPI(
    title="Job Search Ranking API",
    description="BM25 + LightGBM Job Search Engine",
    version="1.0"
)

# Enable React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading data...")

# Load jobs
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

print("API Ready!")


@app.get("/")
def home():
    return {
        "message": "Job Search Ranking API Running"
    }


@app.get("/search")
def search_jobs(q: str):

    query_tokens = q.lower().split()

    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query_tokens)

    candidate_indices = np.argsort(
        bm25_scores
    )[::-1][:100]

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

    # LightGBM Ranking
    ranking_scores = ranker.predict(X)

    reranked = sorted(
        zip(candidate_indices, ranking_scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    seen_titles = set()

    for idx, score in reranked:

        title = df.iloc[idx]["title"]

        if title in seen_titles:
            continue

        seen_titles.add(title)

        results.append({
            "job_id": int(df.iloc[idx]["job_id"]),
            "title": title,
            "company": df.iloc[idx]["company_name"],
            "location": df.iloc[idx]["location"],
            "rank_score": float(score)
        })

        if len(results) == 10:
            break

    return results