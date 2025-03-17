import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

import dataset
from misc import utils
from models.classifier_v1 import ProteinDNAClassifier_v1
from datasets.ecoli_old import ProteinDNADataset

from datetime import datetime

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
        model, dataloader, criterion, optimizer, scheduler, global_iter, device="cuda"
):
    model.train()
    running_loss = 0.0

    for i, (protein, dna, labels) in enumerate(dataloader):
        labels = labels.to(device)
        protein = protein.to(device)
        dna = dna.to(device)

        outputs = model(dna, protein)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # 学习率调度器步进
        scheduler.step()

        running_loss += loss.item()

        global_iter += 1  # 更新全局迭代次数

        # 打印训练日志
        if global_iter % 500 == 0:
            logger.info(
                f"Iteration {global_iter}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}"
            )

        # 返回全局迭代次数，确保同步
        yield global_iter, running_loss / (i + 1)

def validate_model(model, dataloader, criterion, device="cuda"):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for protein, dna, labels in dataloader:
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

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    return val_loss / len(dataloader), accuracy


def main():
    batch_size = 1
    learning_rate = 0.005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    protein_path = "data/ecoli_TF.faa"
    protein_data = dataset.protein_data_preprocess(protein_path)

    dna_path = "data/ecoli_BSTF_with_BSneighborhood.csv"
    dna_data = dataset.dna_data_preprocess(dna_path)

    # 数据分割
    (protein_train, protein_val, train_idx, test_idx) = utils.train_test_split_dict(
        protein_data, dna_data, test_size=0.2
    )

    # 数据加载器
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    dnabert2_model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    ).to(device)
    dnabert2_model.eval()

    train_dataset = ProteinDNADataset(
        protein_train, train_idx, dna_data, DNAbert2=dnabert2_model, DNA_tokenizer=tokenizer
    )
    val_dataset = ProteinDNADataset(
        protein_val, test_idx, dna_data, DNAbert2=dnabert2_model, DNA_tokenizer=tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = ProteinDNAClassifier_v1(768, 1280, 768).to(device)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    # 设置训练迭代参数
    max_iters = 50000
    validate_interval = 5

    best_val_loss = float("inf")
    best_accuracy = 0.0
    best_model_path = f"{log_dir}/binding_resnet_best.pth"
    global_iter = 0

    logger.info("Starting training...")

    # 训练主循环
    while global_iter < max_iters:
        for global_iter, train_loss in train_model(
                model, train_loader, criterion, optimizer, scheduler, global_iter, device
        ):
            # 每隔 validate_interval 次迭代验证一次模型
            if global_iter % validate_interval == 0:
                val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

                logger.info(
                    f"[Validation] Iteration: {global_iter}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                )

                # 如果表现更好，则保存当前最优模型
                if val_loss < best_val_loss and val_accuracy > best_accuracy:
                    best_val_loss = val_loss
                    best_accuracy = val_accuracy

                    torch.save(model.state_dict(), best_model_path)
                    logger.info(
                        f"Model improved. Saving best model with Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                    )

            # 提前终止
            if global_iter >= max_iters:
                break

    logger.info("Training finished.")

    # 保存最终模型
    final_model_path = f"{log_dir}/binding_resnet_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}.")




if __name__ == "__main__":
    main()

