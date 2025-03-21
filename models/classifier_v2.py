import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        """
        基于 nn.MultiheadAttention 的 Self-Attention 模块实现
        :param dim: 输入特征的维度
        :param heads: 多头注意力的头数
        :param dropout: Dropout 比例
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads

        # 使用 PyTorch 原生 MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)

        # 输出投影后的线性层（可以省略，原生实现已经融合了输出投影）
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)  # 添加 LayerNorm，可以选择性启用

    def forward(self, x):
        """
        前向传播
        :param x: 输入，形状 [batch_size, seq_len, dim]
        :return: 自注意力后的特征
        """

        attn_output, _ = self.attention(query=x, key=x, value=x)  # 所有输入均为 x，自注意力

        # 输出投影 + 残差连接 + LayerNorm（标准Transformer结构中常见）
        x = self.layer_norm(x + self.dropout(attn_output))
        return x

# BidirectionalAttention (替换 CrossAttention)
class BiCrossAttention(nn.Module):
    def __init__(self, dna_dim=768, protein_dim=1152, hidden_dim=1152, num_heads=8):
        super(BiCrossAttention, self).__init__()
        self.dna_proj = nn.Linear(dna_dim, hidden_dim)
        # self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        self.cross_attn_dna = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_protein = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, dna_emb, protein_emb):
        dna_proj = self.dna_proj(dna_emb) # (batch, 1, hidden_dim)
        # protein_proj = self.protein_proj(protein_emb).unsqueeze(1)  # (batch, 1, hidden_dim)
        protein_proj = protein_emb  # (batch, 1, hidden_dim)

        attn_dna, _ = self.cross_attn_dna(dna_proj, protein_proj, protein_proj)
        attn_protein, _ = self.cross_attn_protein(protein_proj, dna_proj, dna_proj)

        fused_emb = torch.cat([attn_dna, attn_protein], dim=-1)  # (batch, 1, hidden_dim * 2)
        return fused_emb.squeeze(1)  # (batch, hidden_dim * 2)


# Updated ResBlock with 3 FC layers
class Resblock_3FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Resblock_3FC, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.norm1 = nn.LayerNorm(dim_in)
        self.self_attention = SelfAttention(dim_in)  # 每个 ResBlock 前添加 SelfAttention
        self.fc2 = nn.Linear(dim_in, dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.fc3 = nn.Linear(dim_in, dim_out)
        self.norm3 = nn.LayerNorm(dim_out)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # SelfAttention 前置模块
        x = self.self_attention(x)

        # 三层全连接 + 残差结构
        residual = x
        out = self.norm1(self.fc1(x))
        out = self.relu(out)
        out = self.norm2(self.fc2(out))
        out = self.relu(out)
        out = self.norm3(self.fc3(out))
        return self.relu(out + residual)  # 残差连接


# Updated ProteinDNAClassifier_v2
class ProteinDNAClassifier_v2(nn.Module):
    def __init__(self, input_dim_seq1=768, input_dim_seq2=1152, hidden_dim=1152, num_heads=8):
        super(ProteinDNAClassifier_v2, self).__init__()
        self.dim1 = input_dim_seq1
        self.dim2 = input_dim_seq2
        self.hidden_dim = hidden_dim

        # Self-Attention for each sequence
        self.self_attention_seq1 = SelfAttention(self.dim1)
        self.self_attention_seq2 = SelfAttention(self.dim2)

        # Bidirectional Attention
        self.cross_attention = BiCrossAttention(self.dim1, self.dim2, self.hidden_dim, num_heads)

        # Updated ResBlocks (each prefixed with a self-attention layer)
        self.resblock1a = Resblock_3FC(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.resblock2a = Resblock_3FC(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.resblock3a = Resblock_3FC(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.resblock4a = Resblock_3FC(2 * self.hidden_dim, 2 * self.hidden_dim)

        # Classification Head (4 Linear layers)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2),
            nn.Linear(384, 96),
            nn.LayerNorm(96),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(96, 1)
        )

    def forward(self, dna, tf):
        '''
        
        :param dna: dna,[batch, 768]
        :param tf: protein, [batch, 1280
        :return: 
        '''
        # Self-Attention
        dna = self.self_attention_seq1(dna.unsqueeze(dim=1))
        tf = self.self_attention_seq2(tf.unsqueeze(dim=1))

        # Bidirectional Attention
        x = self.cross_attention(dna, tf)

        # ResBlocks
        x = self.resblock1a(x)
        x = self.resblock2a(x)
        x = self.resblock3a(x)
        x = self.resblock4a(x)

        logits = self.classifier(x).squeeze(-1)
        return logits
