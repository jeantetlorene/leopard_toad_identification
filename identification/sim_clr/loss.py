import torch
import torch.nn as nn
import numpy as np


class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), k=self.batch_size)
        l2 = np.eye((2 * self.batch_size), k=-self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        return (1 - mask).type(torch.bool)

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            2 * self.batch_size, 1
        )
        negative_samples = sim[self.mask].reshape(2 * self.batch_size, -1)

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return self.criterion(logits, labels) / (2 * self.batch_size)
