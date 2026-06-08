import pandas as pd

df = pd.read_csv("clean_jobs.csv")

demo_df = df.sample(
    n=5000,
    random_state=42
)

demo_df.to_csv(
    "clean_jobs_render.csv",
    index=False
)