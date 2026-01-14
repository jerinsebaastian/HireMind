import csv
import random

with open('data/readiness_training_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gap_score', 'readiness'])

    for _ in range(1000):
        gap = random.randint(0, 50)

        if gap < 10:
            readiness = "Job Ready"
        elif gap < 20:
            readiness = "Almost Ready"
        else:
            readiness = "Not Ready"

        writer.writerow([gap, readiness])

print("Training dataset generated successfully.")
