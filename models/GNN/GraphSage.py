import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphSageLinkPredictor(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        
        self.fc1 = nn.Linear(embed_dim * 2 + 2, 64) # +2 for Price, Rate
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_idx, item_idx, features):
        u_emb = self.user_embed(user_idx)
        i_emb = self.item_embed(item_idx)
        
        # Sage-like: Concat user and item info
        x = torch.cat([u_emb, i_emb, features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x