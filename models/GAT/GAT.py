import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGATLinkPredictor(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        self.fc_start = nn.Linear(2, embed_dim) # Project explicit features
        self.fc_out = nn.Linear(embed_dim * 2, 1) # Combine att result + original
        
    def forward(self, user_idx, item_idx, features):
        u_emb = self.user_embed(user_idx) 
        i_emb = self.item_embed(item_idx) 
        f_emb = self.fc_start(features) 
        
        # Attention Mechanism
        q = self.W_q(u_emb).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(i_emb).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(i_emb).view(-1, self.num_heads, self.head_dim)
        
        # Scaled Dot-Product Attention
        attn_scores = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1) 
        
        # Apply attention to value
        attended = (attn_weights * v).view(-1, self.embed_dim)
        
        # Combine User context + Attended Item info
        x = torch.cat([u_emb + f_emb, attended], dim=1)
        output = torch.sigmoid(self.fc_out(x))
        return output