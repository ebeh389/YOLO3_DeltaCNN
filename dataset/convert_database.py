import os
import pandas as pd

rows = []

for file in os.listdir("valid/images"):
    if file.endswith((".jpg", ".jpeg", ".png")):
        label = os.path.splitext(file)[0] + ".txt"
        rows.append([file, label])

pd.DataFrame(rows).to_csv(
    "valid.csv",
    index=False,
    header=["image", "label"]
)