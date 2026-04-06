import sys
sys.path.append('Model')
sys.path.append('models')

from MultiEMO_Model import MultiEMO
from multiemo_graph import MultiEMO_Graph


def build_model_graph(config, *args, **kwargs):
    graph_type = kwargs.pop('graph_type', 'none')
    use_graph = graph_type in ('temporal', 'speaker', 'both')
    if hasattr(config, 'use_graph'):
        use_graph = getattr(config, 'use_graph', use_graph)

    if use_graph:
        return MultiEMO_Graph(config, *args, graph_type=graph_type, **kwargs)
    return MultiEMO(config, *args, **kwargs)
