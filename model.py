import torch
import torch.nn as nn
import torch.nn.functional as F

class BiCrossAttention(nn.Module):
    def __init__(self, dna_dim=768, protein_dim=960, hidden_dim=960, num_heads=8):
        super(BiCrossAttention, self).__init__()
        self.dna_proj = nn.Linear(dna_dim, hidden_dim)
        # self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        self.cross_attn_dna = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_protein = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, dna_emb, protein_emb):
        dna_proj = self.dna_proj(dna_emb).unsqueeze(1)  # (batch, 1, hidden_dim)
        # protein_proj = self.protein_proj(protein_emb).unsqueeze(1)  # (batch, 1, hidden_dim)
        protein_proj = protein_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        attn_dna, _ = self.cross_attn_dna(dna_proj, protein_proj, protein_proj)
        attn_protein, _ = self.cross_attn_protein(protein_proj, dna_proj, dna_proj)
        
        fused_emb = torch.cat([attn_dna, attn_protein], dim=-1)  # (batch, 1, hidden_dim * 2)
        return fused_emb.squeeze(1)  # (batch, hidden_dim * 2)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Reshape if needed (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        if len(x.shape) == 2:
            x_reshaped = x.unsqueeze(1)
        else:
            x_reshaped = x
            
        # Self-attention
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        
        # Add residual connection and norm
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x_reshaped + attn_output)
        
        # Reshape back if needed
        if len(x.shape) == 2:
            output = output.squeeze(1)
            
        return output

class PoolingLayer(nn.Module):
    def __init__(self, pool_type='max'):
        super(PoolingLayer, self).__init__()
        self.pool_type = pool_type
        
    def forward(self, x):
        # If x is 2D (batch_size, features), return as is
        if len(x.shape) == 2:
            return x
            
        # For 3D input (batch_size, seq_len, features)
        if self.pool_type == 'max':
            return torch.max(x, dim=1)[0]
        elif self.pool_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pool_type == 'attention':
            # Simple attention pooling
            attn_weights = torch.softmax(torch.sum(x, dim=-1, keepdim=True), dim=1)
            return torch.sum(x * attn_weights, dim=1)

class AdaptivePoolingLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AdaptivePoolingLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.ones(3))
        self.max_pool = PoolingLayer('max')
        self.mean_pool = PoolingLayer('mean')
        self.attn_pool = PoolingLayer('attention')
        
    def forward(self, x):
        weights = F.softmax(self.attention_weights, dim=0)
        max_pooled = self.max_pool(x)
        mean_pooled = self.mean_pool(x)
        attn_pooled = self.attn_pool(x)
        
        return weights[0] * max_pooled + weights[1] * mean_pooled + weights[2] * attn_pooled

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.bn3 = nn.BatchNorm1d(input_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out += residual
        return self.activation(out)

class DNAProteinClassifier(nn.Module):
    def __init__(self, dna_dim=768, protein_dim=960, hidden_dim=960, num_heads=8):
        super(DNAProteinClassifier, self).__init__()
     
        # Add feature extraction layers before cross-attention
        self.dna_feature_extractor = nn.Sequential(
            nn.Linear(dna_dim, dna_dim),
            nn.LayerNorm(dna_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.protein_feature_extractor = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.LayerNorm(protein_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.bi_cross_attn = BiCrossAttention(dna_dim, protein_dim, hidden_dim, num_heads)
           
        # Pooling layer after cross-attention
        self.pool = PoolingLayer(pool_type='max')
        
        # Self-attention and ResNet blocks
        self.self_attn1 = SelfAttentionBlock(hidden_dim * 2, num_heads)
        self.res_block1 = ResNetBlock(hidden_dim * 2, 960)
        # self.pool1 = AdaptivePoolingLayer()
        
        self.self_attn2 = SelfAttentionBlock(hidden_dim * 2, num_heads)
        self.res_block2 = ResNetBlock(hidden_dim * 2, 960)
        # self.pool2 = AdaptivePoolingLayer()
        
        self.self_attn3 = SelfAttentionBlock(hidden_dim * 2, num_heads)
        self.res_block3 = ResNetBlock(hidden_dim * 2, 960)
        self.pool3 = PoolingLayer(pool_type='mean')
        
        # Gradually reduce dimensions in FC layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Linear(96, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dna_emb, protein_emb):
        dna_features = self.dna_feature_extractor(dna_emb)
        protein_features = self.protein_feature_extractor(protein_emb)
        
        fused_emb = self.bi_cross_attn(dna_features, protein_features)
        fused_emb = self.pool(fused_emb)

        # fused_emb = self.bi_cross_attn(dna_emb, protein_emb)
        # fused_emb = self.pool(fused_emb)
        
        # First block with self-attention
        attn1 = self.self_attn1(fused_emb)
        res1 = self.res_block1(attn1)
        # pool1 = self.pool1(res1)
        
        # Second block with self-attention
        attn2 = self.self_attn2(res1)
        res2 = self.res_block2(attn2)
        # pool2 = self.pool2(res2)
        
        # Third block with self-attention
        attn3 = self.self_attn3(res2)
        res3 = self.res_block3(attn3)
        pool3 = self.pool3(res3)
        
        output = self.fc(pool3)
        return output.squeeze(1)