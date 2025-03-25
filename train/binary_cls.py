import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def train_model(
        model, dataloader, criterion, optimizer, schedular, device="cuda", print_interval=100
):
    model.train()  # 模型进入训练模式
    running_loss = 0.0  # 累积损失

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for i, (protein, dna, labels) in enumerate(progress_bar):
        # 将数据加载到设备上
        labels = labels.to(device)
        protein = protein.to(device)
        dna = dna.to(device)

        optimizer.zero_grad()
        outputs = model(dna, protein)
        loss = criterion(outputs, labels.to(torch.float32))

        # 反向传播 + 优化
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        schedular.step()

        running_loss += loss.item()
        batch_loss = running_loss / (i + 1)
        progress_bar.set_postfix({"Batch Loss": batch_loss})

    return batch_loss


def validate_model(
        model, dataloader, criterion, device="cuda"
):
    model.eval()  # 模型进入验证模式
    correct, total = 0, 0
    true_labels = []
    pred_probs = []

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=True)
        for i, (protein, dna, labels) in enumerate(progress_bar):
            labels = labels.to(device)
            protein = protein.to(device)
            dna = dna.to(device)

            outputs = model(dna, protein)

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_probs.extend(outputs.cpu().numpy())
    # 计算总的 precision 和 recall
    accuracy = correct / total
    auc_score = roc_auc_score(true_labels, pred_probs)

    return accuracy, auc_score
