import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerGraphBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, speaker_ids):
        # features: [N, D]
        # speaker_ids: [N]
        N = features.shape[0]
        device = features.device

        speaker_ids = speaker_ids.long()
        same_speaker = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        idx = torch.arange(N, device=device)
        temporal = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().eq(1).float()

        adj = same_speaker + temporal
        print("Graph nodes:", features.shape[0])
        print("Edge density:", adj.sum().item() / (N * N))

        degree = adj.sum(dim=1).clamp(min=1.0)
        deg_inv_sqrt = degree.pow(-0.5)
        adj_hat = adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)

        return adj_hat


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, h, adj_hat):
        # h: [N, D], adj_hat: [N, N]
        h = torch.matmul(adj_hat, h)
        return self.linear(h)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gcn1 = GraphConvolutionLayer(input_dim)
        self.gcn2 = GraphConvolutionLayer(input_dim)

    def forward(self, h, adj_hat):
        h = F.relu(self.gcn1(h, adj_hat))
        h = F.relu(self.gcn2(h, adj_hat))
        return h
