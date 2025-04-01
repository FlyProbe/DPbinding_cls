import csv
import os

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.manifold import TSNE
import umap

from misc import utils
from tqdm import tqdm
######### Data loading #########

# pt form
# data_path = 'data/tfbs_ESMC.pt'
# data = torch.load(data_path)
# pos_pairs = []
# for i in list(range(len(data))):
#     TF_embedding = data[i]['TF_embedding'].squeeze(0)
#     name =  data[i]['name']
#     for j in list(range(len(data[i]['BS_seq']))):
#         pos_pairs.append({'TF_embedding': TF_embedding.squeeze(0),
#                           'BS_seq': data[i]['BS_seq'][j],
#                           'name': name})


# delete Homo sapiens and Mus musculus
csv_file = pd.read_csv('data/positive_set_v3.csv')
hum_and_mus_list = []
TF = csv_file["TF name"].tolist()
species = csv_file["species"].tolist()
for tf, sp in zip(TF, species):
    if sp == 'Homo sapiens' or sp == 'Mus musculus':
        hum_and_mus_list.append(tf)
hum_and_mus_list = list(set(hum_and_mus_list))
data_path = 'data/tfbs_ESMC.pt'
data = torch.load(data_path)


######### Init Models #########
# tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
# DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
# DNA_model.eval()

# ######### TF Clustering #########
names = [data[i]['name'] for i in range(len(data)) if data[i]['name'] not in hum_and_mus_list]
TF_embedding = [data[i]['TF_embedding'] for i in range(len(data)) if data[i]['name'] not in hum_and_mus_list]
TF_embedding_np = np.stack([tensor.squeeze().cpu().numpy() for tensor in TF_embedding])
# HDBSCAN 聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_epsilon=0.3)
labels_hdbscan = clusterer.fit_predict(TF_embedding_np)
# 统计聚类结果
n_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
print(f"Cluster founded: {n_clusters}")


# 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(TF_embedding_np)

# 绘制 t-SNE 结果
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_hdbscan, cmap="Set1", alpha=0.7)
plt.colorbar(label="Cluster Labels")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of HDBSCAN Clusters")
plt.show()


# 3D
# t-SNE 降维到 3D
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne_3D = tsne.fit_transform(TF_embedding_np)

# 绘制 3D 散点图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 使用聚类标签上色
scatter = ax.scatter(X_tsne_3D[:, 0], X_tsne_3D[:, 1], X_tsne_3D[:, 2],
                     c=labels_hdbscan, cmap="Set1", alpha=0.7)

# 颜色条
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Cluster Labels")

# 设置坐标轴标签
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.set_title("3D t-SNE Visualization of HDBSCAN Clusters")

plt.show()
pass
######### BS Clustering #########
# names = [data[i]['name'] for i in range(len(data))]
mode = 'max'
# BS_embedding = []
# if not os.path.exists(f'data/tfbs_{mode}_DNA.pt'):
#     for pair in tqdm(pos_pairs):
#         dna_eb = utils.DNAbert2_embedding(pair['BS_seq'], tokenizer, DNA_model)
#         BS_embedding.append(dna_eb)
#     torch.save(BS_embedding, f'data/tfbs_{mode}_DNA.pt')
# else:
#     BS_embedding = torch.load(f'data/tfbs_{mode}_DNA.pt')
# BS_embedding_np = np.stack([tensor.cpu().numpy() for tensor in BS_embedding])
# # HDBSCAN 聚类
# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean', cluster_selection_epsilon=1.0)
# labels_hdbscan = clusterer.fit_predict(BS_embedding_np)
# # 统计聚类结果
# n_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
# print(f"Cluster founded: {n_clusters}")

pass
######### CSV BS Clustering #########
# csv form
# data_path = 'data/positive_set_v3_delHUMAN.csv'
# data = pd.read_csv(data_path)
# species = data["species"].tolist()
# TF = data["TF name"].tolist()
# BS = data["binding site sequence"].tolist()
# BS_embedding = []
# pt_path = f'data/delHUMAN_{mode}_DNA.pt'
# if not os.path.exists(pt_path):
#     for bs in tqdm(BS):
#         dna_eb = utils.DNAbert2_embedding(bs, tokenizer, DNA_model)
#         BS_embedding.append(dna_eb)
#     torch.save(BS_embedding, pt_path)
# else:
#     BS_embedding = torch.load(pt_path)
# BS_embedding_np = np.stack([tensor.cpu().numpy() for tensor in BS_embedding])
# # HDBSCAN 聚类
# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean', cluster_selection_epsilon=1.0)
# labels_hdbscan = clusterer.fit_predict(BS_embedding_np)
# # 统计聚类结果
# n_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
# print(f"Cluster founded: {n_clusters}")



# df = pd.DataFrame({'cluster': labels_hdbscan, 'BS':BS, 'TF':TF})
# df.to_csv(f"cluster_check/BS_delHUMAN_clusters_{mode}.csv", index=False)

