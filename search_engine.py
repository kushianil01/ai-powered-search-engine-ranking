import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading jobs...")

df = pd.read_csv("clean_jobs.csv")

# Combine searchable text
df["search_text"] = (
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["title"].fillna("") + " " +
    df["skills_desc"].fillna("") + " " +
    df["description"].fillna("")
)

print("Creating search index...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1,2)
)

job_vectors = vectorizer.fit_transform(df["search_text"])

print("Search engine ready!")

while True:

    query = input("\nSearch jobs: ")

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        job_vectors
    ).flatten()

    top_indices = scores.argsort()[-10:][::-1]

    print("\nTop Results:\n")

    for rank, idx in enumerate(top_indices, start=1):

        print(f"{rank}. {df.iloc[idx]['title']}")
        print(f"   Company: {df.iloc[idx]['company_name']}")
        print(f"   Location: {df.iloc[idx]['location']}")
        print(f"   Score: {scores[idx]:.4f}")
        print()