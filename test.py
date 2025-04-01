import csv
import json
import pandas as pd

path = r"cluster_check/BS_delHUMAN_clusters_max.csv"
data = pd.read_csv(path, sep=",")
TF = data["TF"].tolist()
clusters = data["cluster"].tolist()

res = {}
for tfid, cluster in zip(TF, clusters):
    if tfid in res:
        if cluster in res[tfid]:
            res[tfid][cluster] = res[tfid][cluster] + 1
        else:
            res[tfid][cluster] = 1
    else:
        res[tfid] = {cluster:1}

with open("cluster_check/BSmean_TFmean_Clustering_Check_delHUMAN.json", "w") as f:
    json.dump(res, f, indent=4)
