import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DETR(nn.Module):
    def __init__(self, num_classes=13, num_queries=100, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()
        self.transformer = Transformer(num_encoder_layers, hidden_dim, num_heads, hidden_dim * 4, num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.panoptic_head = PanopticHead(hidden_dim, num_classes, mask_dim=1)

    def forward(self, src, mask):
        bs, c, h, w = src.size()
        src = src.flatten(2).permute(2, 0, 1)  # (hw, bs, c)
        pos_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # positional embeddings
        tgt = torch.zeros_like(pos_embed)
        hs = self.transformer(src, tgt + pos_embed, self.query_embed.weight)
        class_logits, masks = self.panoptic_head(hs[-1])
        return class_logits, masks.view(bs, h, w)
    
class Transformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_queries=100, ff_dim=2048, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

    def forward(self, src, query_embed, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(self.things_stuff_query, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return memory, output

class PanopticHead(nn.Module):
    def __init__(self, feature_dim, num_classes, mask_dim):
        super(PanopticHead, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, mask_dim, kernel_size=1)
        )
    
    def forward(self, features):
        classes = self.classifier(features)
        masks = self.mask_predictor(features.unsqueeze(-1).unsqueeze(-1))
        return classes, masks
