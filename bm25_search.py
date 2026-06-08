import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

print("Loading jobs...")

# Load cleaned dataset
df = pd.read_csv("clean_jobs.csv")

# Remove duplicate jobs
df = df.drop_duplicates(subset=["job_id"])

print(f"Jobs after deduplication: {len(df)}")

# Fill missing values
df = df.fillna("")

# Give title extra weight
df["search_text"] = (
    df["title"] + " " +
    df["title"] + " " +
    df["title"] + " " +
    df["skills_desc"] + " " +
    df["description"]
)

print("Building BM25 index...")

# Tokenize corpus
corpus = [
    doc.lower().split()
    for doc in df["search_text"]
]

# Build BM25 model
bm25 = BM25Okapi(corpus)

print("BM25 Search Ready!")

while True:

    query = input("\nSearch jobs: ")

    if query.lower() == "exit":
        break

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:10]

    print("\nTop Results:\n")

    for rank, idx in enumerate(top_indices, start=1):

        print(f"{rank}. {df.iloc[idx]['title']}")
        print(f"   Company: {df.iloc[idx]['company_name']}")
        print(f"   Location: {df.iloc[idx]['location']}")
        print(f"   Score: {scores[idx]:.2f}")
        print()