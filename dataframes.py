import pandas as pd

df_linkedin = pd.read_csv("data/merged_linkedin_jobs.csv")
df_techmap = pd.read_csv("data/techmap-jobs_us_2023-05-05.csv")
df_fake = pd.read_csv("data/Fake Postings.csv")
df_fake2 = pd.read_csv("data/fake_job_postings.csv")

print(f"LinkedIn shape: {df_linkedin.shape}")
print(f"Techmap shape: {df_techmap.shape}")
print(f"Fake jobs shape: {df_fake.shape}")
print(f"Fake jobs 2 shape: {df_fake2.shape}")