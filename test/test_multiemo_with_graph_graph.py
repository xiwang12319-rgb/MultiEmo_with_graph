import sys
import os

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from models.graph_module import SpeakerGraphBuilder, GraphEncoder
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

    model = MultiEMO_Graph(
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

    graph_builder = SpeakerGraphBuilder()
    graph_encoder = GraphEncoder(model_dim * 3)

    try:
        model.eval()
        with torch.no_grad():
            text_feat, audio_feat, visual_feat, speaker_ids_out = model.encoder(texts, audios, visuals, speaker_ids)
            graph_input = torch.cat([text_feat, audio_feat, visual_feat], dim=-1)
            graph_input_flat = graph_input.reshape(-1, graph_input.shape[-1])
            speaker_ids_flat = speaker_ids_out.reshape(-1)

            adj = graph_builder(graph_input_flat, speaker_ids_flat)
            graph_feat = graph_encoder(graph_input_flat, adj)
            enhanced_feat = graph_input_flat + graph_feat
            enhanced_feat = enhanced_feat.reshape(seq_len, batch, -1)
            logits = model.fusion(enhanced_feat)

        print("graph_input shape:", graph_input.shape)
        print("adjacency matrix:\n", adj)
        print("graph_feat shape:", graph_feat.shape)
        print("enhanced_feat shape:", enhanced_feat.shape)
        print("logits shape:", logits.shape)
        print("logits head:", logits[:3].tolist())
        print("Conclusion: graph forward success")
    except Exception as exc:
        print("Conclusion: graph forward failed")
        print("Error:", repr(exc))


if __name__ == "__main__":
    main()
