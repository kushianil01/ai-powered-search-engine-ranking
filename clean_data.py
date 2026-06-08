import pandas as pd

print("Loading dataset...")

df = pd.read_csv("archive/postings.csv")

print("Original shape:", df.shape)

df = df[
    [
        "job_id",
        "company_name",
        "title",
        "description",
        "location",
        "formatted_work_type",
        "formatted_experience_level",
        "skills_desc",
        "remote_allowed"
    ]
]

print("Columns selected")

df = df.dropna(subset=["title", "description"])

print("After dropping null title/description:", df.shape)

df = df.fillna("")

df.to_csv("clean_jobs.csv", index=False)

print("Saved clean_jobs.csv successfully")