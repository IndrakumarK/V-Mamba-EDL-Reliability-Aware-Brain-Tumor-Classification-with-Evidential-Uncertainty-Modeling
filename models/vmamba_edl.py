import torch.nn as nn
from .vision_mamba import VisionMambaEncoder
from .evidential_head import EvidentialHead


class VMambaEDL(nn.Module):
    def __init__(self, num_classes=4, embed_dim=768):
        super().__init__()
        self.encoder = VisionMambaEncoder(embed_dim=embed_dim)
        self.edl_head = EvidentialHead(embed_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        edl_out = self.edl_head(z)
        return z, edl_out
