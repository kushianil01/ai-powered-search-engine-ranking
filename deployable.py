import pandas as pd

df = pd.read_csv("clean_jobs.csv")

demo_df = df.sample(
    n=20000,
    random_state=42
)

demo_df.to_csv(
    "clean_jobs_demo.csv",
    index=False
)

print(demo_df.shape)