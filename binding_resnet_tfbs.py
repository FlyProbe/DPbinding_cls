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
        model, dataloader, criterion, optimizer, device="cuda", print_interval=100

):
    model.train()  # 模型进入训练模式
    running_loss = 0.0  # 累积损失
    running_corrects = 0  # 用于计算准确率的正确预测数量
    total_samples = 0  # 总样本数
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


        probs = torch.sigmoid(outputs)  # 转换为概率
        preds = (probs >= 0.5).int()

        running_corrects += (preds == labels).sum().item()  # 累计正确的预测
        total_samples += labels.size(0)  # 累计样本总数

        # 每 print_interval 次打印一次运行结果
        if (i + 1) % print_interval == 0:
            batch_acc = running_corrects / total_samples  # 当前准确率
            batch_loss = running_loss / (i + 1)  # 当前平均损失
            print(
                f"Iteration [{i + 1}/{num_batches}]: "
                f"Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}"
            )

        # 更新 tqdm 的描述信息
        progress_bar.set_postfix({"Batch Loss": loss.item()})



    # 计算一个 epoch 的平均损失
    average_loss = running_loss / num_batches

    return average_loss


def validate_model(model, dataloader, criterion, device="cuda"):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing", leave=True)
        for protein, dna, labels in progress_bar:
            labels = labels.to(device)
            protein = protein.to(device)
            dna = dna.to(device)

            outputs = model(dna, protein)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            all_labels.extend(labels.int().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            progress_bar.set_postfix({"Batch Loss": loss.item()})

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    return val_loss / len(dataloader), accuracy


def main():
    # 超参数
    online = False
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 20
    validation_split = 0.2  # 验证集比例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    if online:
        # 在线加载数据（保留之前的逻辑）
        data_path = 'data/tfbs_dataset_with_negatives.csv'
        full_data = pd.read_csv(data_path).to_dict(orient='records')
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        DNA_model.eval()
        ESM_model.eval()

        dataset = TFBSWithNeg(
            full_data,
            DNAbert2=DNA_model,
            DNA_tokenizer=tokenizer,
            ESM_model=ESM_model,
            ESM_alphabet=alphabet
        )
    else:
        # 离线加载数据
        BS_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_DNA.pt"
        TF_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_ESMC.pt"

        BS = torch.load(BS_path)
        TF = torch.load(TF_path)

        data = [{"BS_embedding": bs['BS_embedding'],
                 "TF_embedding": tf["TF_embedding"],
                 "label": torch.argmax(bs["label"]).to(torch.float32)} for bs, tf in zip(BS, TF)]

        # dataset = TFBSWithNeg_offline(data)

    # 将正样本和负样本分开
    pos = [sample for sample in data if sample["label"] == 1]  # 样本标签为 [0, 1] 是正样本
    neg = [sample for sample in data if sample["label"] == 0]  # 样本标签为 [1, 0] 是负样本

    num_neg_samples = round(len(pos)*2)
    if len(neg) > num_neg_samples:
        neg = random.sample(neg, num_neg_samples)

    # 合并正负样本，构造新的数据集
    data = pos + neg
    random.shuffle(data)  # 打乱样本顺序

    # 划分训练集和验证集
    num_val = int(len(data) * validation_split)
    val_data = data[:num_val]  # 验证集
    train_data = data[num_val:]  # 训练集

    # 构造 DataLoader
    train_subset = TFBSWithNeg_offline(train_data)
    val_subset = TFBSWithNeg_offline(val_data)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = ProteinDNAClassifier_v1(768, 1152, 768).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from transformers import get_cosine_schedule_with_warmup
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # num_warmup_steps = 1
    # num_training_steps = len(train_loader) * num_epochs
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps * len(train_loader),  # warm-up steps: 按批次总数计算
    #     num_training_steps=num_training_steps
    # )


    best_val_loss = float("inf")
    best_accuracy = 0.0

    # 开始单独训练
    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}...")

        # 训练模型
        train_loss = train_model(
            model, train_loader, criterion, optimizer, device
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


def test(model_path, output_acc=True, batch_size=1, device="cuda"):
    # 确认设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集加载过程
    try:
        online = False  # 更改此处支持在线和离线数据加载
        if online:
            print("Loading dataset in online mode...")
            # 在线加载数据
            data_path = 'data/tfbs_dataset_with_negatives.csv'
            full_data = pd.read_csv(data_path).to_dict(orient='records')
            tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
            DNA_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
            ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

            DNA_model.eval()
            ESM_model.eval()

            dataset = TFBSWithNeg(
                full_data,
                DNAbert2=DNA_model,
                DNA_tokenizer=tokenizer,
                ESM_model=ESM_model,
                ESM_alphabet=alphabet
            )

        else:
            print("Loading dataset in offline mode...")
            # 离线加载数据
            BS_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_DNA.pt"
            TF_path = r"D:\projects\DPbinding_cls\data\tfbs_dataset_with_negatives_ESMC.pt"

            BS = torch.load(BS_path)
            TF = torch.load(TF_path)

            data = [{"BS_embedding": bs['BS_embedding'],
                     "TF_embedding": tf["TF_embedding"],
                     "label": torch.argmax(bs["label"]).to(torch.float32)} for bs, tf in zip(BS, TF)]
            dataset = TFBSWithNeg_offline(data)

        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 构造测试数据的 DataLoader
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    print("Initializing model...")
    model = ProteinDNAClassifier_v1(768, 1152, 768).to(device)

    def kaiming_normal_initialize(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):  # 适用于 Conv2d 和 Linear 层
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming 正态分布
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 把偏置初始化为 0

    model.apply(kaiming_normal_initialize)

    # 加载模型权重
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

    results = []  # 用于保存预测结果
    with torch.no_grad():  # 禁用梯度计算
        print("Starting inference...")
        for batch in tqdm(test_loader):
            protein, dna, labels = batch

            # 将数据加载到设备
            protein = protein.to(device)
            dna = dna.to(device)

            # 前向推理
            outputs = model(dna, protein)  # 获得 logits
            predictions = torch.argmax(outputs, dim=1)  # 获取概率最大的类别

            # 将预测结果保存到列表中
            # results.extend(predictions.cpu().tolist())

            # 如果需要计算准确率
            if output_acc:
                for label, prediction in zip(labels, predictions):
                    label = torch.argmax(label)
                    label = label.item()
                    prediction = prediction.item()

                    # 累计总样本数与正确预测数
                    total_samples[label] += 1
                    if label == prediction:
                        correct_predictions[label] += 1

    print("Inference complete.")

    if output_acc:
        accuracy_dict = {}
        for label in total_samples:
            accuracy = correct_predictions[label] / total_samples[label]
            accuracy_dict[label] = accuracy
        print("Per-label Accuracy:", accuracy_dict)
        print(correct_predictions)
        return results, accuracy_dict

    return results, None



if __name__ == "__main__":
    main()

    # # infer
    # model_path = r"log\20250312_171815\fold_1_best_model.pth"
    #
    # # 调用测试函数
    # predictions, acc = test(model_path)
    #
    # # 打印预测结果
    # print("ACC:", acc)


