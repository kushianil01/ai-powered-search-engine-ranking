import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

print("Loading jobs...")

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
    "backend developer",
    "software engineer",
    "frontend developer",
    "data scientist",
    "java developer",
    "cloud engineer",
    "devops engineer"
]

rows = []

for query in queries:

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:20]

    for rank, idx in enumerate(top_indices):

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

        # Auto relevance label
        if rank < 5:
            relevance = 3
        elif rank < 10:
            relevance = 2
        else:
            relevance = 1

        rows.append({
            "query": query,
            "job_id": df.iloc[idx]["job_id"],
            "title": df.iloc[idx]["title"],
            "bm25_score": scores[idx],
            "title_matches": title_matches,
            "desc_matches": desc_matches,
            "skill_matches": skill_matches,
            "relevance": relevance
        })

ltr_df = pd.DataFrame(rows)

ltr_df.to_csv(
    "ltr_dataset.csv",
    index=False
)

print(ltr_df.head())
print("\nSaved ltr_dataset.csv")