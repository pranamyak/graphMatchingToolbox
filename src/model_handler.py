import src.dataset as dataset

from src.models.simgnn import SimGNN
from src.models.graphsim import GraphSim
from src.models.gotsim import GOTSim
from src.models.eric import ERIC
from src.models.egsc import EGSC

model_name_to_class_mappings = {
    'simgnn': SimGNN,
    'graphsim': GraphSim,
    'gotsim': GOTSim,
    'eric': ERIC,
    'egsc': EGSC
}

def get_model_names():
    return list(model_name_to_class_mappings.keys())

def get_model(model_name, config, max_node_set_size, max_edge_set_size, device):
    if model_name.startswith('gmn_baseline'):
        model_class = GMNBaseline
    elif model_name.startswith('gmn_iterative_refinement'):
        model_class = GMNIterativeRefinement
    elif model_name.startswith('edge_early_interaction_baseline_1'):
        model_class = EdgeEarlyInteractionBaseline1
    elif model_name.startswith('edge_early_interaction_1'):
        model_class = EdgeEarlyInteraction1
    elif model_name.startswith('isonet'):
        model_class = ISONET
    else:
        model_class = model_name_to_class_mappings[model_name]

    return model_class(
        max_node_set_size=max_node_set_size,
        max_edge_set_size=max_edge_set_size,
        device=device,
        **config
    )

def get_data_type_for_model(model_name):
    if model_name in ['simgnn', 'graphsim', 'gotsim', 'eric', 'egsc']:
        return dataset.PYG_DATA_TYPE
    return dataset.GMN_DATA_TYPE
