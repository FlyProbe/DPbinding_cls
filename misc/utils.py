import os
import sys
import random
import numpy as np
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

def cal_reverse_complement(seq):
    seq =  list(seq.upper())
    lut = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_seq = seq[::-1]
    for i, base in enumerate(rev_seq):
        rev_seq[i] = lut.get(base, 'N')
    seq = "".join(seq)
    rev_seq = "".join(rev_seq)
    res = seq if seq < rev_seq else rev_seq
    return res

def train_test_split_dict(protein_data, dna_data, test_size=0.2):
    train_size = 1 - test_size
    keys = list(protein_data.keys() & dna_data.keys())
    random.shuffle(keys)

    # 计算第一个字典中键的数量
    split_index = int(len(keys) * train_size)

    # 构建两个字典
    train_idx = keys[:split_index]
    test_idx = keys[split_index:]
    train_data = {key: protein_data[key] for key in train_idx}
    test_data = {key: protein_data[key] for key in test_idx}

    return train_data, test_data, train_idx, test_idx

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

def DNAbert2_embedding(dna_seq, tokenizer, model):
    '''

    :param dna_seq: String, dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    :param tokenizer: See https://github.com/MAGICS-LAB/DNABERT_2
    :param model: See https://github.com/MAGICS-LAB/DNABERT_2
    :return: Tensor, shape = (768,)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    tokens = tokenizer(dna_seq, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(tokens)[0]
        # embedding = torch.mean(outputs[0], dim=0)
        embedding = torch.max(outputs[0], dim=0)[0]
    return embedding

def ESM2_embedding(protein_Seq, model, alphabet):
    '''

    :param protein_Seq: List of tuple, [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")]
    :param model: See https://github.com/facebookresearch/esm
    :param alphabet: See https://github.com/facebookresearch/esm
    :return: List of tensor, shape = (1280,)
    '''
    # TODO: Out of memory on my computer
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    model.eval()
    batch_converter = alphabet.get_batch_converter()

    res = []
    for protein_name, cur_seq in protein_Seq:
        batch_labels, batch_strs, batch_tokens = batch_converter([(protein_name, cur_seq)])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
        res.append(sequence_representations[0])
    return res


if __name__ == '__main__':
    import torch
    import esm
    from transformers import AutoTokenizer, AutoModel
    from transformers.models.bert.configuration_bert import BertConfig

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(
        "cuda")

    ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    protein = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]

    dna_eb = DNAbert2_embedding(dna, tokenizer, DNA_model)
    protein_eb = ESM2_embedding(protein, ESM_model, alphabet)
    pass



