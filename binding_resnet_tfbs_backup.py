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
from sklearn.model_selection import train_test_split

import esm

from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

from collections import Counter

from models.classifier_v1 import ProteinDNAClassifier_v1
from datasets.tfbs_with_neg import TFBSWithNeg, TFBSWithNeg_offline

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


def train_model(
        model, dataloader, criterion, optimizer, device="cuda"
):
    model.train()  # 模型进入训练模式
    running_loss = 0.0  # 累积损失
    num_batches = len(dataloader)  # 总批次数

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for i, (protein, dna, labels) in enumerate(progress_bar):

        # 将数据加载到设备上
        labels = labels.to(device)
        protein = protein.to(device)
        dna = dna.to(device)

        # 前向传播
        outputs = model(dna, protein)
        loss = criterion(outputs, labels)

        # 反向传播 + 优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 累加损失
        running_loss += loss.item()

        # 更新 tqdm 的描述信息（实时显示训练损失）
        progress_bar.set_postfix({"Batch Loss": running_loss})


    # 计算一个 epoch 的平均损失
    average_loss = running_loss / num_batches

    return average_loss


def validate_model(model, dataloader, criterion, device="cuda"):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Training", leave=True)
        for protein, dna, labels in progress_bar:
            labels = labels.to(device)
            protein = protein.to(device)
            dna = dna.to(device)

            outputs = model(dna, protein)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            _, gt = torch.max(labels, 1)

            all_labels.extend(gt.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            progress_bar.set_postfix({"Batch Loss": val_loss})

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    return val_loss / len(dataloader), accuracy


def main():
    online = False
    batch_size = 1
    learning_rate = 0.005
    num_epochs = 20  # 训练 epoch 的数量
    validate_interval = 1  # 每个 epoch 验证一次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    if online:
        data_path = 'data/tfbs_dataset_with_negatives.csv'
        full_data = pd.read_csv(data_path).to_dict(orient='records')

        train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

        # 初始化数据处理器
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

        ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        DNA_model.eval()
        ESM_model.eval()

        train_dataset = TFBSWithNeg(
            train_data,
            DNAbert2=DNA_model,
            DNA_tokenizer=tokenizer,
            ESM_model=ESM_model,
            ESM_alphabet=alphabet
        )
        val_dataset = TFBSWithNeg(
            test_data,
            DNAbert2=DNA_model,
            DNA_tokenizer=tokenizer,
            ESM_model=ESM_model,
            ESM_alphabet=alphabet
        )
    else:
        BS_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_DNA.pt"
        TF_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_ESMC.pt"

        BS = torch.load(BS_path)
        TF = torch.load(TF_path)

        data = [{"BS_embedding": bs['BS_embedding'],
                 "TF_embedding": tf["TF_embedding"],
                 "label": bs["label"]} for bs, tf in zip(BS, TF)]
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_dataset = TFBSWithNeg_offline(train_data)
        val_dataset = TFBSWithNeg_offline(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    # dim2 = 1280 for ESM2, 1152 for ESMC
    model = ProteinDNAClassifier_v1(768, 1152, 768).to(device)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建学习率调度器（CosineAnnealingLR）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练日志相关
    best_val_loss = float("inf")
    best_accuracy = 0.0
    best_model_path = f"{log_dir}/binding_resnet_tfbs_best.pth"

    logger.info("Starting training...")

    # 开始训练
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs} starting...")

        # 训练模型
        train_loss = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # 验证模型
        if (epoch + 1) % validate_interval == 0:
            val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

            logger.info(
                f"[Validation] Epoch: {epoch + 1}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
            )

            # 保存最佳模型
            if val_loss < best_val_loss and val_accuracy > best_accuracy:
                best_val_loss = val_loss
                best_accuracy = val_accuracy

                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"Model improved. Saving best model with Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                )

        # 调度器步进
        scheduler.step()  # 根据 epoch 更新学习率

        # 打印当前学习率
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} finished, Training Loss: {train_loss:.4f}, Current LR: {current_lr}")

    logger.info("Training finished.")

    # 保存最终模型
    final_model_path = f"{log_dir}/binding_resnet_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}.")


if __name__ == "__main__":
    main()

