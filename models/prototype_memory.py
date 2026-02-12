import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))

    def update(self, features, labels):
    for k in torch.unique(labels):
        mask = labels == k
        class_feat = features[mask].mean(dim=0)

        # Initialize if prototype is zero (first update)
        if torch.norm(self.prototypes[k]) == 0:
            self.prototypes[k] = class_feat
        else:
            self.prototypes[k] = (
                self.momentum * self.prototypes[k]
                + (1.0 - self.momentum) * class_feat
            )


    def similarity(self, z):
        z_norm = F.normalize(z, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        return torch.matmul(z_norm, proto_norm.t())
