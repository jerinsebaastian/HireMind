import random
import pandas as pd

data = []

for _ in range(2000):  # generate 2000 samples

    # Simulate skill levels (5 skills, 0–3)
    skills = [random.randint(0, 3) for _ in range(5)]

    # Simulate weights (1–5)
    weights = [random.randint(1, 5) for _ in range(5)]

    REQUIRED_LEVEL = 3

    total_gap = 0
    missing = 0
    weak = 0
    moderate = 0
    strong = 0
    high_importance_gap = 0

    for level, weight in zip(skills, weights):
        gap = max(0, REQUIRED_LEVEL - level)
        weighted_gap = gap * weight
        total_gap += weighted_gap

        if level == 3:
            strong += 1
        elif level == 2:
            moderate += 1
        elif level == 1:
            weak += 1
        else:
            missing += 1

        if weight >= 4:
            high_importance_gap += weighted_gap

    avg_skill_level = sum(skills) / len(skills)

    # Logical labeling rule (for training supervision)
    if total_gap < 5 and missing == 0:
        readiness = "Job Ready"
    elif total_gap < 15:
        readiness = "Almost Ready"
    else:
        readiness = "Not Ready"

    data.append([
        total_gap,
        missing,
        weak,
        moderate,
        strong,
        avg_skill_level,
        high_importance_gap,
        readiness
    ])

columns = [
    "total_gap",
    "missing_count",
    "weak_count",
    "moderate_count",
    "strong_count",
    "avg_skill_level",
    "high_importance_gap",
    "readiness"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("data/multifeature_training_data.csv", index=False)

print("Multi-feature training dataset generated successfully.")