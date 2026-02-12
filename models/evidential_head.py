import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        evidence = F.softplus(self.fc(z))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)

        belief = evidence / S
        uncertainty = alpha.shape[1] / S

        return {
            "evidence": evidence,
            "alpha": alpha,
            "belief": belief,
            "uncertainty": uncertainty.squeeze(1),
        }
