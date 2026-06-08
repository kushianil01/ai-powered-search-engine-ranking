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

queries = [
    "python backend intern",
    "machine learning engineer",
    "data engineer",
    "software engineer",
    "backend developer",
    "frontend developer",
    "java developer",
    "data scientist",
    "cloud engineer",
    "devops engineer"
]

training_rows = []

for query in queries:

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:20]

    for idx in top_indices:

        training_rows.append({
            "query": query,
            "job_id": df.iloc[idx]["job_id"],
            "title": df.iloc[idx]["title"],
            "bm25_score": scores[idx],
            "relevance": ""
        })

training_df = pd.DataFrame(training_rows)

training_df.to_csv(
    "training_data.csv",
    index=False
)

print("Saved training_data.csv")
print(training_df.head())