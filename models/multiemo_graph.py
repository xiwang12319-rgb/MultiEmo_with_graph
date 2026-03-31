import sys
sys.path.append('Model')

import torch
import torch.nn as nn
import torch.nn.functional as F

from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP
from graph_module import SpeakerGraphBuilder, GraphEncoder


class MultiEMO_Graph(nn.Module):
    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers,
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag
        self.model_dim = model_dim
        self.n_speakers = n_speakers

        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)

        self.graph_builder = SpeakerGraphBuilder()
        self.graph_encoder = GraphEncoder(model_dim * 3)

        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(model_dim * 3, model_dim)

        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

    def _normalize_speaker_ids(self, speaker_ids, seq_len, batch):
        if speaker_ids.dim() == 1:
            speaker_ids = speaker_ids.unsqueeze(1).expand(-1, batch)
        return speaker_ids

    def encoder(self, texts, audios, visuals, speaker_ids):
        seq_len, batch, _ = texts.shape
        speaker_ids = self._normalize_speaker_ids(speaker_ids, seq_len, batch)
        speaker_masks = F.one_hot(speaker_ids.long(), num_classes=self.n_speakers).float()
        utterance_masks = torch.ones(batch, seq_len, device=texts.device, dtype=texts.dtype)

        text_features = self.text_fc(texts)
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)
        audio_features = self.audio_fc(audios)
        audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)
        visual_features = self.visual_fc(visuals)
        visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

        return text_features, audio_features, visual_features, speaker_ids

    def fusion(self, fusion_input):
        text_features = fusion_input[..., :self.model_dim]
        audio_features = fusion_input[..., self.model_dim:self.model_dim * 2]
        visual_features = fusion_input[..., self.model_dim * 2:self.model_dim * 3]

        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)
        if self.multi_attn_flag == True:
            fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(text_features, audio_features, visual_features)
        else:
            fused_text_features, fused_audio_features, fused_visual_features = text_features, audio_features, visual_features

        fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
        fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
        fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])

        fused_features = torch.cat((fused_text_features, fused_audio_features, fused_visual_features), dim=-1)
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)

        return mlp_outputs

    def forward(self, texts, audios, visuals, speaker_ids):
        text_feat, audio_feat, visual_feat, speaker_ids = self.encoder(texts, audios, visuals, speaker_ids)

        graph_input = torch.cat([text_feat, audio_feat, visual_feat], dim=-1)
        seq_len, batch, dim = graph_input.shape
        graph_input_flat = graph_input.reshape(-1, dim)
        speaker_ids_flat = speaker_ids.reshape(-1)

        adj = self.graph_builder(graph_input_flat, speaker_ids_flat)
        graph_feat = self.graph_encoder(graph_input_flat, adj)
        enhanced_feat = graph_input_flat + graph_feat
        enhanced_feat = enhanced_feat.reshape(seq_len, batch, dim)

        output = self.fusion(enhanced_feat)
        return output
