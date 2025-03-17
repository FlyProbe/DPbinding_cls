import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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