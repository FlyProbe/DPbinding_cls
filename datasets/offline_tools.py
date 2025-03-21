'''
ESM-C needs python >= 3.10
DNAbert2 needs python <= 3.8
'''

import os
import json
from misc import utils

import pandas as pd
import torch


from tqdm import tqdm

def load_data(path):
    data = pd.read_csv(path).to_dict(orient='records')
    return data

def generate_offline_dataset_DNA(path):
    from transformers import AutoTokenizer, AutoModel
    from transformers.models.bert.configuration_bert import BertConfig

    fn = os.path.splitext(os.path.split(path)[-1])[0] + '_DNA.pt'
    dst = os.path.join(os.path.split(path)[0], fn)
    if os.path.exists(dst):
        os.remove(dst)

    data = load_data(path)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DNA_model.to(device)
    DNA_model.eval()

    res = []

    for i in tqdm(range(len(data))):
        dna_seq = data[i].get('binding site sequence')
        if pd.isna(dna_seq):
            continue

        tokens = tokenizer(dna_seq, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            outputs = DNA_model(tokens)[0]
            embedding = torch.mean(outputs[0], dim=0)
        label = data[i].get('label')
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).float()


        res.append({'id': i, 'BS_embedding': embedding, 'label': label_one_hot})

    torch.save(res, dst)

def generate_offline_dataset_ESMC(path):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    fn = os.path.splitext(os.path.split(path)[-1])[0] + '_ESMC.pt'
    dst = os.path.join(os.path.split(path)[0], fn)
    if os.path.exists(dst):
        os.remove(dst)

    data = load_data(path)
    res = []
    client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

    for i in tqdm(range(len(data))):
        protein = ESMProtein(sequence=data[i].get('TF sequence'))
        if pd.isna(protein):
            continue

        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embedding = torch.mean(logits_output.embeddings, dim=1)

        label = data[i].get('label')
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).float()
        res.append({'id': i, 'TF_embedding': embedding, 'label': label_one_hot})

    torch.save(res, dst)

def regroup_with_TF(**kwargs):
    path = kwargs.get('path')
    if isinstance(path, list):
        data = []
        for p in path:
            data.extend(load_data(p))
    elif isinstance(path, str):
        data = load_data(path)

    data = [it for it in data if it['label'] == 1]
    res = {}
    for it in data:
        if it['TF sequence'] in res:
            res[it['TF sequence']].append(it['binding site sequence'])
        else:
            res[it['TF sequence']] = [it['TF name'],it['binding site sequence']]
    return res

def ESMC_json_protein(path):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    fn = os.path.splitext(os.path.split(path)[-1])[0] + '_ESMC.pt'
    dst = os.path.join(os.path.split(path)[0], fn)
    if os.path.exists(dst):
        os.remove(dst)

    with open(path, 'r') as f:
        data = json.load(f)
    res = []
    client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"

    for k in tqdm(list(data.keys())):
        protein = ESMProtein(sequence=k)

        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embedding = torch.mean(logits_output.embeddings, dim=1)

        res.append({'name': data[k][0], 'TF_embedding': embedding, 'BS_seq': data[k][1:]})

    torch.save(res, dst)


if __name__ == '__main__':
    # path = r'D:\projects\ProteinDNABinding\ProteinDNABinding\py\data\tfbs_dataset_with_negatives.csv'
    # # python <= 3.8
    # generate_offline_dataset_DNA(path)
    # # python >= 3.10
    # # generate_offline_dataset_ESMC(path)

    # path  = [r'../data/training_dataset_with_negatives_v4.csv',
    #          r'../data/test_dataset_with_negatives_v4.csv']
    # res = regroup_with_TF(path=path)
    # with open(r'../data/tfbs.json', 'w') as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)

    ESMC_json_protein(r"D:\projects\DPbinding_cls\data\tfbs.json")

