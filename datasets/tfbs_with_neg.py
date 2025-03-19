import sys
import os

import torch
from torch.utils.data import Dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from misc import utils
from datasets.transforms import dna_aug


class TFBSWithNeg(Dataset):
    def __init__(self,
                 data,
                 pos_r=0.3,
                 neg_r=0.6,
                 rand_r=0.1,
                 **kwargs):
        self.data = data

        self.DNAbert2 = kwargs.get("DNAbert2", None)
        self.DNA_tokenizer = kwargs.get("DNA_tokenizer", None)
        self.ESM_model = kwargs.get("ESM_model", None)
        self.ESM_alphabet = kwargs.get("ESM_alphabet", None)

        # TODO: NOT USED. Currently just use a fixed dataset.
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.rand_r = rand_r

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        TF_sqeuence = self.data[idx]['TF sequence']  # Protein
        binding_site = self.data[idx]['binding site sequence']  # DNA
        label = self.data[idx]['label']
        dna_eb = utils.DNAbert2_embedding(binding_site, self.DNA_tokenizer, self.DNAbert2)
        protein_eb = utils.ESM2_embedding([('protein', TF_sqeuence)], self.ESM_model, self.ESM_alphabet)

        dna_eb = dna_eb
        protein_eb = protein_eb[0]
        tmp = [0, 0]
        tmp[label] = 1
        label = torch.tensor(tmp, dtype=torch.float32)

        return (protein_eb,
                dna_eb,
                label)

class TFBSWithNeg_offline(Dataset):
    def __init__(self,
                 data,
                 pos_r=0.3,
                 neg_r=0.6,
                 rand_r=0.1,
                 **kwargs):
        self.data = data

        # TODO: NOT USED. Currently just use a fixed dataset.
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.rand_r = rand_r

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_eb = self.data[idx]['TF_embedding'].squeeze(0)
        dna_eb = self.data[idx]['BS_embedding']
        label = self.data[idx]['label']

        return (protein_eb,
                dna_eb,
                label)


class TFBSWithNeg_flexDNA_TESTONLY(Dataset):
    def __init__(self,
                 data,
                 pos_r=0.3,
                 neg_r=0.6,
                 rand_r=0.1,
                 **kwargs):
        self.data = data
        self.extension = kwargs.get("extension", None)
        self.DNAbert2 = kwargs.get("DNAbert2", None)
        self.DNA_tokenizer = kwargs.get("DNA_tokenizer", None)

        # TODO: NOT USED. Currently just use a fixed dataset.
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.rand_r = rand_r

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_eb = self.data[idx]['TF_embedding'].squeeze(0)
        dna = dna_aug.DNA_extension(self.data[idx]['BS_seq'], self.extension)
        dna_eb = utils.DNAbert2_embedding(dna, self.DNA_tokenizer, self.DNAbert2)
        label = self.data[idx].get('label', None)

        return (protein_eb,
                dna_eb,
                label)



if __name__ == '__main__':
    pass




