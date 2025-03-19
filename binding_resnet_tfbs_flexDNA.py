import logging
import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

import esm

from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

from collections import Counter

from models.classifier_v1 import ProteinDNAClassifier_v1
from datasets.tfbs_with_neg import TFBSWithNeg, TFBSWithNeg_offline, TFBSWithNeg_flexDNA_TESTONLY
from train.binary_cls import train_model, validate_model

from datetime import datetime
from tqdm import tqdm

# 获取当前时间戳，并格式化为字符串
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(f"log/{timestamp}")
log_dir = f"log/{timestamp}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/training_{timestamp}.log", mode="w"),
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)


def test(model_path, output_acc=True, batch_size=1, device="cuda", **kwargs):
    csv_file_path = kwargs.get("csv_file_path", None)
    # 确认设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = 'data/tfbs_dataset_with_negatives.csv'
    full_data = pd.read_csv(data_path).to_dict(orient='records')
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    DNA_model.eval()

    TF_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_ESMC.pt"
    TF = torch.load(TF_path)

    data = [{'BS_seq':bs['binding site sequence'],
             'TF_embedding':tf['TF_embedding'] ,
             'label': bs['label']}
            for tf, bs in zip(TF, full_data)]

    # transforms
    extension = kwargs.get("extension", 0)
    dataset = TFBSWithNeg_flexDNA_TESTONLY(
        data[:1000],
        pos_r=0.3,
        neg_r=0.6,
        rand_r=0.1,
        DNAbert2=DNA_model,
        DNA_tokenizer=tokenizer,
        extension=extension
    )

    print("Dataset loaded successfully.")


    # 构造测试数据的 DataLoader
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    print("Initializing model...")
    model = ProteinDNAClassifier_v1(768, 1152, 768).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # 切换模型为评估模式
    model.eval()

    # 推理阶段
    from collections import defaultdict

    total_samples = defaultdict(int)  # 每个类别的样本总数
    correct_predictions = defaultdict(int)  # 每个类别的正确预测数

    results = []
    total_correct = 0  # 总正确数
    total_count = 0  # 总样本数

    with torch.no_grad():  # 禁用梯度计算
        print("Starting inference...")
        for batch in tqdm(test_loader):
            protein, dna, labels = batch

            # 将数据加载到设备
            protein = protein.to(device)
            dna = dna.to(device)

            # 前向推理
            outputs = model(dna, protein)  # 获得 logits
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            probs = probs.unsqueeze(0) if probs.ndim == 0 else probs
            preds = preds.unsqueeze(0) if preds.ndim == 0 else preds

            for i in range(len(preds)):
                results.append({'cls_pred': preds[i].item(), 'confidence_of_match': probs[i].item()})

                if output_acc:
                    # for label, prediction in zip(labels, preds):
                    label = labels[i].item()  # 转换为 Python 标量
                    pred = preds[i].item()  # 转换为 Python 标量

                    # 累计总样本数与正确预测数
                    total_samples[label] += 1
                    total_count += 1
                    if label == pred:
                        correct_predictions[label] += 1
                        total_correct += 1

    print("Inference complete.")

    # todo: save all the printed infos into csv
    # todo: batch

    if output_acc:
        # 计算总准确率和分类准确率
        cls_acc = []
        overall_accuracy = total_correct / total_count
        print(f"Overall Accuracy: {overall_accuracy:.4f}")  # 输出总准确率
        print(f"Total Correct: {total_correct} / {total_count}")  # 总正确数和总样本数

        # 计算每个类别的分类准确率和分类正确数
        for label in sorted(total_samples.keys()):  # 按类别排序输出
            class_accuracy = correct_predictions[label] / total_samples[label] if total_samples[label] > 0 else 0.0
            cls_acc.append(class_accuracy)
            print(
                f"Class {label} Accuracy: {class_accuracy:.4f} ({correct_predictions[label]} / {total_samples[label]})")
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([extension, overall_accuracy, *cls_acc])

    return results



if __name__ == "__main__":

    # infer
    model_path = r"log/20250314_172258/best_model.pth"

    output_csv_path = os.path.join("log", "extnsion_fixed_length_1.csv")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Extension", "Overall Accuracy", "Class Accuracy_1", "Class Accuracy_2"])

    # 调用测试函数
    for i in range(20):
        predictions = test(model_path, batch_size=1, extension=i+5000, csv_file_path=output_csv_path)





