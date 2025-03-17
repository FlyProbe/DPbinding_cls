import torch
from torch.utils.data import Dataset
import random
import numpy as np

from collections import Counter


def generate_random_dna_sequence(dna_seq):
    """
    根据给定的 DNA 序列生成一段比例相同的随机 DNA 序列。

    :param dna_seq: 原始 DNA 序列，只包含 'A', 'T', 'C', 'G'
    :return: 生成的随机 DNA 序列，同长度且字母比例相同
    """
    # 统计字母比例
    counts = Counter(dna_seq)
    total = len(dna_seq)

    # 计算每个字母的比例
    weights = [counts['A'] / total, counts['T'] / total, counts['C'] / total, counts['G'] / total]

    # 按比例生成随机序列
    random_sequence = random.choices(population=['A', 'T', 'C', 'G'], weights=weights, k=total)

    # 返回为字符串
    return ''.join(random_sequence)

class ProteinDNADataset(Dataset):
    def __init__(self, protein_embeddings, id, dna_embeddings, pos_r=0.3, neg_r=0.6, rand_r=0.1, **kwargs,):
        self.protein_embeddings = protein_embeddings
        self.dna_embeddings = dna_embeddings
        # assert pos_r + neg_r + rand_r == 1
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.rand_r = rand_r
        self.id = id

        self.DNAbert2 = kwargs.get("DNAbert2", None)
        self.DNA_tokenizer = kwargs.get("DNA_tokenizer", None)

    def __len__(self):
        return len(self.protein_embeddings)

    # p = [self.pos_r, self.neg_r, self.rand_r]
    def __getitem__(self, idx):
        data_type = np.random.choice(["pos", "neg", "rand"], p=[self.pos_r, self.neg_r, self.rand_r])
        id = self.id[idx]
        protein_embedding = self.protein_embeddings[id][0]
        if data_type == "pos":
            dna_embedding = random.choice(self.dna_embeddings[id][1:])
            label = [1,0] # True, False

        elif data_type == "neg":
            ng_idx = np.random.randint(len(self.protein_embeddings))
            while ng_idx == idx:
                ng_idx = np.random.randint(len(self.protein_embeddings))

            dna_embedding = random.choice(self.dna_embeddings[self.id[ng_idx]][1:])
            label = [0,1]
        else:
            assert self.DNAbert2 is not None and self.DNA_tokenizer is not None, \
                "set rand_r = 0 if there is no DNAbert2 or DNA_tokenizer"
            # try:
            #     dna_seq = random.choice(self.dna_embeddings[id][0])
            # except:
            #   dna_seq = random.choice(self.dna_embeddings[random.choice(list(self.dna_embeddings.keys()))][0])
            dna_seq = random.choice(self.dna_embeddings[id][0])
            fake_dna_seq = generate_random_dna_sequence(dna_seq)

            tokens = self.DNA_tokenizer(fake_dna_seq, return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                outputs = self.DNAbert2(tokens)[0]
                dna_embedding = torch.mean(outputs[0], dim=0)

            label = [0,1]

        return (protein_embedding,  # 1280,
                dna_embedding,  # 768,
                torch.tensor(label, dtype=torch.float32))
