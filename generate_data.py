
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

df = pd.DataFrame({
    "daily_screen_hours": np.random.uniform(0.5, 12, n),
    "app_opens_per_day": np.random.randint(10, 180, n),
    "social_media_hours": np.random.uniform(0, 8, n),
    "night_usage_hours": np.random.uniform(0, 5, n),
    "sleep_hours": np.random.uniform(3, 10, n),
    "control_difficulty": np.random.randint(1, 6, n)
})

# Illustrative synthetic labels, not clinical assessments.
score = (
    (df["daily_screen_hours"] > 6).astype(int)
    + (df["app_opens_per_day"] > 90).astype(int)
    + (df["social_media_hours"] > 4).astype(int)
    + (df["night_usage_hours"] > 2).astype(int)
    + (df["sleep_hours"] < 6).astype(int)
    + (df["control_difficulty"] >= 4).astype(int)
)

df["risk_level"] = np.select(
    [score <= 1, score <= 3],
    ["Low", "Moderate"],
    default="High"
)

df.to_csv("app_usage_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nRisk distribution:")
print(df["risk_level"].value_counts())