import sys
import os

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.graph_module import SpeakerGraphBuilder, GraphEncoder


def build_adj_raw(speaker_ids, temporal=True, same_speaker=True):
    N = speaker_ids.shape[0]
    device = speaker_ids.device

    adj = torch.zeros(N, N, device=device)
    if same_speaker:
        same = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
        adj = adj + same
    if temporal:
        idx = torch.arange(N, device=device)
        temporal_adj = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().eq(1).float()
        adj = adj + temporal_adj
    return adj


def normalize_adj(adj):
    degree = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = degree.pow(-0.5)
    return adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)


def run_ablation(name, graph_input, speaker_ids, temporal, same_speaker):
    print(f"\n--- Ablation: {name} ---")
    adj_raw = build_adj_raw(speaker_ids, temporal=temporal, same_speaker=same_speaker)
    adj_hat = normalize_adj(adj_raw)
    encoder = GraphEncoder(graph_input.shape[-1])
    graph_feat = encoder(graph_input, adj_hat)
    enhanced_feat = graph_input + graph_feat

    print("adjacency matrix:\n", adj_raw)
    print("edge density:", adj_raw.sum().item() / (adj_raw.shape[0] * adj_raw.shape[0]))
    print("neighbors per node:", adj_raw.sum(dim=1).tolist())
    print("graph_feat head:\n", graph_feat[:2])

    return enhanced_feat


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 6
    D = 8
    speaker_ids = torch.tensor([0, 1, 0, 1, 0, 1], device=device)
    graph_input = torch.randn(N, D, device=device)

    builder = SpeakerGraphBuilder()
    encoder = GraphEncoder(D)

    adj = builder(graph_input, speaker_ids)
    graph_feat = encoder(graph_input, adj)
    enhanced_feat = graph_input + graph_feat

    print("adjacency matrix:\n", adj)
    print("edge density:", adj.sum().item() / (N * N))
    print("neighbors per node:", adj.sum(dim=1).tolist())
    print("graph_feat:\n", graph_feat)

    print("\nNode 0 check:")
    print("node 0 before:", graph_input[0].tolist())
    print("node 0 after:", enhanced_feat[0].tolist())
    neighbors = (build_adj_raw(speaker_ids) [0] > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
    print("node 0 neighbors (indices):", neighbors)
    print("node 0 expected neighbors: temporal (1) + same-speaker (2,4)")

    enhanced_temporal = run_ablation(
        "temporal-only",
        graph_input,
        speaker_ids,
        temporal=True,
        same_speaker=False,
    )
    enhanced_same = run_ablation(
        "same-speaker-only",
        graph_input,
        speaker_ids,
        temporal=False,
        same_speaker=True,
    )

    diff_temporal = (enhanced_temporal - graph_input).abs().mean().item()
    diff_same = (enhanced_same - graph_input).abs().mean().item()

    print("\nSummary:")
    print("temporal edges are effective" if diff_temporal >= 1e-6 else "temporal edges are not effective")
    print("same-speaker edges are effective" if diff_same >= 1e-6 else "same-speaker edges are not effective")


if __name__ == "__main__":
    main()
