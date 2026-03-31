import sys
sys.path.append('Model')
sys.path.append('models')

from MultiEMO_Model import MultiEMO
from multiemo_graph import MultiEMO_Graph


def build_model_graph(config, *args, **kwargs):
    if getattr(config, 'use_graph', False):
        return MultiEMO_Graph(*args, **kwargs)
    return MultiEMO(*args, **kwargs)
