import os
import sys
import torch
from torch.utils.data import Dataset
import random
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from misc import utils
from datasets.transforms import dna_aug

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
            fake_dna_seq = dna_aug.generate_random_dna_sequence(dna_seq)

            tokens = self.DNA_tokenizer(fake_dna_seq, return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                outputs = self.DNAbert2(tokens)[0]
                dna_embedding = torch.mean(outputs[0], dim=0)

            label = [0,1]

        return (protein_embedding,  # 1280,
                dna_embedding,  # 768,
                torch.tensor(label, dtype=torch.float32))
