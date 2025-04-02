import argparse
import csv
import os
import random
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from misc import utils
from tqdm import tqdm


def stability_check(path):
    source = pd.read_csv(path)
    ori_cls = source['cluster'].tolist()
    new_cls = source['new_cluster'].tolist()
    res = {}
    for ori, new in zip(ori_cls, new_cls):
        if ori == -1:
            continue
        else:
            if ori in res:
                res[ori].append(new)
            else:
                res[ori] = [new]
    ori_cls_num = []
    new_cls_num = []
    pencent = []
    new_cls = []
    for key in res:
        o = len(res[key])
        counter = Counter(res[key])
        cls, n = counter.most_common(1)[0]
        ori_cls_num.append(o)
        new_cls_num.append(n)
        pencent.append(n/o)
        new_cls.append(cls)
    return ori_cls_num, new_cls_num, pencent, new_cls

def main():
    snp_ins = f"cluster_check/BS_snp_clusters_ins.csv"
    ori, new, pct, new_cls = stability_check(snp_ins)
    df = pd.DataFrame({'ori': ori, 'new': new, 'percent': pct, 'new_cls': new_cls})
    df.to_csv(f"cluster_check/BS_stability_ins.csv", index=False)

    snp_del = f"cluster_check/BS_snp_clusters_del.csv"
    ori, new, pct, new_cls = stability_check(snp_del)
    df = pd.DataFrame({'ori': ori, 'new': new, 'percent': pct, 'new_cls': new_cls})
    df.to_csv(f"cluster_check/BS_stability_del.csv", index=False)

    snp_mut = f"cluster_check/BS_snp_clusters_mut.csv"
    ori, new, pct, new_cls = stability_check(snp_mut)
    df = pd.DataFrame({'ori': ori, 'new': new, 'percent': pct, 'new_cls': new_cls})
    df.to_csv(f"cluster_check/BS_stability_mut.csv", index=False)
    # 用原分类做字典label，新分类做list，然后用counter来计数，记百分比

if __name__ == '__main__':
    main()


