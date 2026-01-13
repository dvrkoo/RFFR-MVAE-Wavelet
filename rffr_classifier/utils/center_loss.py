import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, feature_dim, margin=2.0):
        super(CenterLoss, self).__init__()
        self.feature_dim = feature_dim
        self.margin = margin

        self.center = nn.Parameter(torch.randn(feature_dim))

    def forward(self, features):
        batch_size = features.size(0)

        center = self.center.unsqueeze(0).expand(batch_size, -1)

        distances = torch.norm(features - center, p=2, dim=1)

        loss = torch.mean(torch.relu(distances - self.margin))

        return loss, distances

    def get_center(self):
        return self.center.detach()

    def compute_anomaly_score(self, features):
        with torch.no_grad():
            batch_size = features.size(0)
            center = self.center.unsqueeze(0).expand(batch_size, -1)
            distances = torch.norm(features - center, p=2, dim=1)
        return distances


class CompactnessLoss(nn.Module):
    def __init__(self, feature_dim, num_classes=1, margin=2.0, repulsion_weight=0.5):
        super(CompactnessLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.repulsion_weight = repulsion_weight

        self.center = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels=None):
        if labels is None:
            labels = torch.zeros(
                features.size(0), dtype=torch.long, device=features.device
            )

        batch_size = features.size(0)

        center = self.center[0].unsqueeze(0).expand(batch_size, -1)

        distances = torch.norm(features - center, p=2, dim=1)

        real_mask = labels == 0
        fake_mask = labels == 1

        if real_mask.sum() > 0:
            compactness_loss = distances[real_mask].mean()
        else:
            compactness_loss = torch.tensor(0.0, device=features.device)

        if fake_mask.sum() > 0:
            repulsion_loss = torch.mean(
                torch.relu(self.margin - distances[fake_mask]) ** 2
            )
        else:
            repulsion_loss = torch.tensor(0.0, device=features.device)

        center_reg = 1 * torch.norm(self.center[0]) ** 2

        loss = compactness_loss + self.repulsion_weight * repulsion_loss + center_reg

        return loss, distances, compactness_loss, repulsion_loss

    def get_center(self, class_idx=0):
        return self.center[class_idx].detach()

    def compute_anomaly_score(self, features, class_idx=0):
        with torch.no_grad():
            batch_size = features.size(0)
            center = self.center[class_idx].unsqueeze(0).expand(batch_size, -1)
            distances = torch.norm(features - center, p=2, dim=1)
        return distances
