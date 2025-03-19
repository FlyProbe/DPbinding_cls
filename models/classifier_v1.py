import torch
import torch.nn as nn
import torch.nn.functional as F


# Cross Attention模块
class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, out_dim, num_heads):
        super(CrossAttention, self).__init__()

        self.relu = nn.ReLU()
        self.seq1_linear = nn.Linear(dim1, out_dim)
        self.seq2_linear = nn.Linear(dim2, out_dim)

        self.multihead_attention = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)

    def forward(self, seq1, seq2):
        """
        Args:
            seq1: Tensor of shape (batch_size, seq_len1, embed_dim) - query
            seq2: Tensor of shape (batch_size, seq_len2, embed_dim) - key & value
        """
        # 将 seq1 和 seq2 的特征维度统一到 out_dim
        seq1 = self.relu(self.seq1_linear(seq1))  # (batch_size, 1, out_dim)
        seq2 = self.relu(self.seq2_linear(seq2))  # (batch_size, seq_len2, out_dim)

        # 注意 MultiheadAttention 要求 inputs：(batch_size, seq_len, embed_dim)
        # seq1 作为 Query，seq2 作为 Key 和 Value
        attn_output, _ = self.multihead_attention(seq1, seq2, seq2)
        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)  # 计算注意力权重

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # 转置 keys 以适配矩阵乘法: (batch_size, dim, seq_len)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len, seq_len)

        # 注意力加权求和
        out = torch.matmul(attention_weights, values)  # (batch_size, seq_len, dim)
        return out


# Res Block模块 (Linear替代conv)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.relu(out + residual)


class ProteinDNAClassifier_v1(nn.Module):
    def __init__(self, dim1, dim2, out_dim, num_heads=8, cls=1):
        super(ProteinDNAClassifier_v1, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.out_dim = out_dim

        # Cross Attention模块
        self.cross_attention = CrossAttention(self.dim1, self.dim2, self.out_dim, num_heads)

        # ResNet部分：四个ResBlock，并在中间插入Pooling操作
        # 每个ResBlock的输入维度随着Pooling操作逐渐减半
        self.resblock1a = ResBlock(self.out_dim)
        self.attention1 = SelfAttention(self.out_dim)

        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.resblock2a = ResBlock(self.out_dim // 2)
        # self.attention2 = SelfAttention(self.out_dim // 2)

        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.resblock3a = ResBlock(self.out_dim // 4)
        # self.attention3 = SelfAttention(self.out_dim // 4)

        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.resblock4a = ResBlock(self.out_dim // 8)
        # self.attention4 = SelfAttention(self.out_dim // 8)

        # 二分类头部
        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim // 8, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, cls)
        )
        # self.norm = nn.LayerNorm(cls)

    def forward(self, seq1, seq2):
        """
        Args:
            seq1: Tensor of shape (batch_size, length), Protein
            seq2: Tensor of shape (batch_size, length), DNA
        Returns:
            logits: Tensor of shape (batch_size, 2)
        """
        seq1 = seq1.unsqueeze(1)  # (batch_size, 1, length)
        seq2 = seq2.unsqueeze(1)  # (batch_size, 1, length)

        fused_seq = self.cross_attention(seq1, seq2)  # (batch_size, 1, out_dim)

        x = self.resblock1a(fused_seq)
        x = self.attention1(x)

        x = self.pool1(x)
        x = self.resblock2a(x)
        # x = self.attention2(x)

        x = self.pool2(x)
        x = self.resblock3a(x)
        # x = self.attention3(x)

        x = self.pool3(x)
        x = self.resblock4a(x)
        # x = self.attention4(x)


        # 最终通过二分类头
        logits = self.classifier(x)  # (batch_size, 1)
        return logits.squeeze()



# 测试模型
if __name__ == "__main__":
    batch_size = 8
    seq1 = torch.randn(batch_size, 768)  # 输入1: (batch_size, 768)
    seq2 = torch.randn(batch_size, 1280)  # 输入2: (batch_size, 1280)

    model = ProteinDNAClassifier_v1()
    print(model)
    output = model(seq1, seq2)
    print(output.shape)  # 输出: (batch_size, 2)
