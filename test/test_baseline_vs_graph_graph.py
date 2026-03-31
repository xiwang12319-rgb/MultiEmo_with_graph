import sys
import os

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from Model.MultiEMO_Model import MultiEMO
from models.multiemo_graph import MultiEMO_Graph


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 6
    batch = 2
    roberta_dim = 8
    D_m_audio = 8
    D_m_visual = 8
    model_dim = 8
    n_classes = 4
    n_speakers = 2

    texts = torch.randn(seq_len, batch, roberta_dim, device=device)
    audios = torch.randn(seq_len, batch, D_m_audio, device=device)
    visuals = torch.randn(seq_len, batch, D_m_visual, device=device)
    speaker_ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], device=device)
    speaker_masks = torch.nn.functional.one_hot(speaker_ids, num_classes=n_speakers).float()
    utterance_masks = torch.ones(batch, seq_len, device=device)
    padded_labels = torch.zeros(seq_len * batch, dtype=torch.long, device=device)

    torch.manual_seed(1)
    baseline_model = MultiEMO(
        dataset='IEMOCAP',
        multi_attn_flag=False,
        roberta_dim=roberta_dim,
        hidden_dim=16,
        dropout=0.0,
        num_layers=2,
        model_dim=model_dim,
        num_heads=2,
        D_m_audio=D_m_audio,
        D_m_visual=D_m_visual,
        D_g=model_dim,
        D_p=model_dim,
        D_e=model_dim,
        D_h=model_dim,
        n_classes=n_classes,
        n_speakers=n_speakers,
        listener_state=False,
        context_attention='simple',
        D_a=model_dim,
        dropout_rec=0.0,
        device=device,
    ).to(device)

    torch.manual_seed(1)
    graph_model = MultiEMO_Graph(
        dataset='IEMOCAP',
        multi_attn_flag=False,
        roberta_dim=roberta_dim,
        hidden_dim=16,
        dropout=0.0,
        num_layers=2,
        model_dim=model_dim,
        num_heads=2,
        D_m_audio=D_m_audio,
        D_m_visual=D_m_visual,
        D_g=model_dim,
        D_p=model_dim,
        D_e=model_dim,
        D_h=model_dim,
        n_classes=n_classes,
        n_speakers=n_speakers,
        listener_state=False,
        context_attention='simple',
        D_a=model_dim,
        dropout_rec=0.0,
        device=device,
    ).to(device)

    baseline_model.eval()
    graph_model.eval()
    with torch.no_grad():
        _, _, _, _, baseline_logits = baseline_model(
            texts, audios, visuals, speaker_masks, utterance_masks, padded_labels
        )
        graph_logits = graph_model(texts, audios, visuals, speaker_ids)

    diff = (baseline_logits - graph_logits).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    print("baseline output head:", baseline_logits[:3].tolist())
    print("graph output head:", graph_logits[:3].tolist())
    print("mean absolute difference:", mean_diff)
    print("max absolute difference:", max_diff)

    if mean_diff < 1e-6:
        print("Conclusion: graph not affecting output")
    else:
        print("Conclusion: graph is affecting output")


if __name__ == "__main__":
    main()
