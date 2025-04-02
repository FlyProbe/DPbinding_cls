import csv
import os
import random
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

# delete Homo sapiens and Mus musculus
csv_file = pd.read_csv('cluster_check/BS_clusters_max.csv')
BS = csv_file["BS"].tolist()
ori_cluster = csv_file["cluster"].tolist()

######### Init Models #########
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
DNA_model.eval()

######### BS Clustering #########
mode = 'mut'

bs_snp = []
bs_embedding = []
bs_SNP_embedding = []
ori_cluster_snp = []

def snp_augment(dna):
    def insert_dna(seq):
        bases = "ATCG"
        insert_pos = random.randint(1, len(seq)-1)  # 允许插入到末尾
        new_base = random.choice(bases)
        mutated_seq = seq[:insert_pos] + new_base + seq[insert_pos:]
        return mutated_seq

    def delete_dna(seq):
        if len(seq) < 2:
            return seq  # 避免删除空序列
        delete_pos = random.randint(1, len(seq) - 2)
        mutated_seq = seq[:delete_pos] + seq[delete_pos + 1:]
        return mutated_seq

    def mutate_dna(seq):
        if len(seq) == 0:
            return seq  # 避免突变空序列
        bases = "ATCG"
        mutate_pos = random.randint(1, len(seq) - 2)
        original_base = seq[mutate_pos]
        new_base = random.choice(bases.replace(original_base, ""))  # 避免突变成自己
        mutated_seq = seq[:mutate_pos] + new_base + seq[mutate_pos + 1:]
        return mutated_seq

    # mode = np.random.choice(['ins', 'del', 'mu'])
    if mode == 'ins':
        dna = insert_dna(dna)
    elif mode == 'del':
        dna = delete_dna(dna)
    elif mode == 'mut':
        dna = mutate_dna(dna)

    return dna

for i, dna in enumerate(tqdm(BS)):
    dna_extend = snp_augment(dna)
    dna_eb = utils.DNAbert2_embedding(dna, tokenizer, DNA_model)
    dna_extend_eb = utils.DNAbert2_embedding(dna_extend, tokenizer, DNA_model)
    bs_snp.extend([dna, dna_extend])
    bs_embedding.append(dna_eb)
    bs_SNP_embedding.append(dna_eb)
    bs_SNP_embedding.append(dna_extend_eb)
    ori_cluster_snp.extend([ori_cluster[i]]*2)

BS_embedding_np = np.stack([tensor.cpu().numpy() for tensor in bs_embedding])
BS_embedding_snp_np = np.stack([tensor.cpu().numpy() for tensor in bs_SNP_embedding])
# # HDBSCAN 聚类
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean', cluster_selection_epsilon=1.0)
# labels_hdbscan = clusterer.fit_predict(BS_embedding_np)
labels_hdbscan_snp = clusterer.fit_predict(BS_embedding_snp_np)
# # 统计聚类结果
# n_clusters = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
# print(f"Cluster founded: {n_clusters}")

pass

df = pd.DataFrame({'cluster': ori_cluster_snp, 'BS_SNP':bs_snp, 'new_cluster':labels_hdbscan_snp})
df.to_csv(f"cluster_check/BS_snp_clusters_{mode}.csv", index=False)

