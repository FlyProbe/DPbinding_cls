import logging
import os

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

from models.classifier_v2 import ProteinDNAClassifier_v2
from datasets.transforms import dna_aug
from datasets.tfbs_with_neg import TFBSWithNeg_flexDNA
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


def main():
    # 超参数
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 200
    validation_split = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extension = 0

    data_path = 'data/tfbs_ESMC.pt'
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    DNA_model.eval()

    data = torch.load(data_path)

    # 初始化模型
    model = ProteinDNAClassifier_v2(768, 1152, 1152).to(device)
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



    best_val_loss = float("inf")
    best_accuracy = 0.0
    # pos pairs are fixed, should be generated outside the loop
    pos_pairs = []
    for i in list(range(len(data))):
        TF_embedding = data[i]['TF_embedding']
        for j in list(range(len(data[i]['BS_seq']))):
            pos_pairs.append({'TF_embedding': TF_embedding,
                              'BS_seq': data[i]['BS_seq'][j],
                              'label': 1})
    random.shuffle(pos_pairs)
    num_val = int(len(pos_pairs) * validation_split)
    neg, rand = 0.8, 0.2


    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}...")
        # generate dataset every 2 epoch
        if epoch % 4 == 0:

            neg_pairs = []
            for i in list(range(len(data))):
                TF_embedding = data[i]['TF_embedding']
                for j in list(range(len(data[i]['BS_seq']))):
                    # generate negative
                    data_type = np.random.choice(["neg", "rand"], p=[neg, rand])
                    if data_type == "neg":
                        fake_idx = np.random.randint(len(data))
                        while fake_idx == i:
                            fake_idx = np.random.randint(len(data))
                        BS_seq = np.random.choice(data[fake_idx]['BS_seq'])
                        neg_pairs.append({'TF_embedding' :TF_embedding,
                                          'BS_seq':BS_seq,
                                          'label': 0})
                    elif data_type == "rand":
                        BS_seq = dna_aug.generate_random_dna_sequence(data[i]['BS_seq'][j])
                        while BS_seq in data[i]['BS_seq']:
                            BS_seq = dna_aug.generate_random_dna_sequence(data[i]['BS_seq'][j])
                        neg_pairs.append({'TF_embedding' :TF_embedding,
                                          'BS_seq':BS_seq,
                                          'label': 0})

        random.shuffle(neg_pairs)
        val_data = pos_pairs[:num_val] + neg_pairs[:num_val]
        train_data = pos_pairs[num_val:] + neg_pairs[num_val:]

        train_subset = TFBSWithNeg_flexDNA(
            train_data,
            pos_r=0.5,
            neg_r=0.4,
            rand_r=0.1,
            DNAbert2=DNA_model,
            DNA_tokenizer=tokenizer,
            extension=extension
        )
        val_subset = TFBSWithNeg_flexDNA(
            val_data,
            pos_r=0.8,
            neg_r=0.1,
            rand_r=0.1,
            DNAbert2=DNA_model,
            DNA_tokenizer=tokenizer,
            extension=extension
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

        # 训练模型
        train_loss = train_model(
            model, train_loader, criterion, optimizer, device, print_interval=100
        )

        # 验证模型
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss and val_accuracy > best_accuracy:
            best_val_loss = val_loss
            best_accuracy = val_accuracy

            torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
            logger.info(f"Best model saved with Val Loss: {best_val_loss:.4f}, "
                        f"Val Accuracy: {best_accuracy:.4f}.")

        # 调整学习率
        scheduler.step()

    # 打印最终验证集上的性能
    logger.info(f"Training Complete. Final Val Loss: {best_val_loss:.4f}, Final Val Accuracy: {best_accuracy:.4f}")
    print(f"Training Complete. Final Val Loss: {best_val_loss:.4f}, Final Val Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()

