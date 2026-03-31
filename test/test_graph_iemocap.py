import sys
import os

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from optparse import OptionParser

from models.graph_module import SpeakerGraphBuilder, GraphEncoder
from Dataset.IEMOCAPDataset import IEMOCAPDataset
from models.multiemo_graph import MultiEMO_Graph


def get_args():
    parser = OptionParser()
    parser.add_option('--batch_size', dest='batch_size', default=5, type='int', help='batch size')
    parser.add_option('--limit_train', dest='limit_train', default=5, type='int', help='limit train samples (0 = no limit)')
    parser.add_option('--limit_test', dest='limit_test', default=0, type='int', help='limit test samples (0 = no limit)')

    (options, _) = parser.parse_args()

    return options


def main():
    args = get_args()
    torch.manual_seed(0)
    device = torch.device('cpu')

    N = 5
    D = 12
    features = torch.randn(N, D)
    speaker_ids = torch.tensor([0, 1, 0, 1, 1])

    builder = SpeakerGraphBuilder()
    adj = builder(features, speaker_ids)
    encoder = GraphEncoder(D)
    graph_feat = encoder(features, adj)

    print("Adj matrix:\n", adj)

    base_train_dataset = IEMOCAPDataset(train=True)
    base_test_dataset = IEMOCAPDataset(train=False)

    train_dataset = base_train_dataset
    if args.limit_train and args.limit_train > 0:
        limit = min(args.limit_train, len(base_train_dataset))
        train_dataset = Subset(base_train_dataset, list(range(limit)))

    test_dataset = base_test_dataset
    if args.limit_test and args.limit_test > 0:
        limit = min(args.limit_test, len(base_test_dataset))
        test_dataset = Subset(base_test_dataset, list(range(limit)))

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base_train_dataset.collate_fn)

    model = MultiEMO_Graph(
        dataset='IEMOCAP',
        multi_attn_flag=True,
        roberta_dim=768,
        hidden_dim=1024,
        dropout=0.0,
        num_layers=2,
        model_dim=256,
        num_heads=4,
        D_m_audio=512,
        D_m_visual=1000,
        D_g=256,
        D_p=256,
        D_e=256,
        D_h=256,
        n_classes=6,
        n_speakers=2,
        listener_state=False,
        context_attention='simple',
        D_a=256,
        dropout_rec=0.0,
        device=device,
    ).to(device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            texts, audios, visuals, speaker_masks, utterance_masks, labels = [x.to(device) for x in batch]
            speaker_ids = torch.argmax(speaker_masks, dim=-1)
            logits = model(texts, audios, visuals, speaker_ids)
            print('Logits shape:', logits.shape)
            break


if __name__ == '__main__':
    main()
