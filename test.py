import csv
import json
import pandas as pd

path = r"cluster_check/BS_delHUMAN_clusters_max.csv"
data = pd.read_csv(path)
TF = data["TF"].tolist()
clusters = data["cluster"].tolist()

res = []
for tf, cls in zip(TF, clusters):
    if cls == 834:
        res.append(tf)

res = list(set(res))
print(res)