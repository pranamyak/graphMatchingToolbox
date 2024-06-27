import torch
import numpy as np
from lap import lapjv
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

def dense_wasserstein_distance_v3(cost_matrix):
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(cost_matrix)
    return np.eye(cost_matrix.shape[0])[col_ind_lapjv]

class GOTSim(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        max_edge_set_size,
        max_node_set_size,
        device,
        filters,
        dropout,
        is_sig,
        gnn_type
    ):
        """
        """
        super(GOTSim, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.device = device
        self.is_sig = is_sig
        self.gnn_type = gnn_type
        self.filters = filters
        self.num_gcn_layers = len(filters)

        # Initialize GNN layers
        self.conv_layers = torch.nn.ModuleList()
        in_channels = self.input_dim
        for out_channels in filters:
            self.conv_layers.append(self.get_gnn_layer(in_channels, out_channels))
            in_channels = out_channels

        # TODO: fix this
        self.n1 = max_node_set_size
        self.n2 = max_node_set_size
        self.insertion_constant_matrix = 99999 * (torch.ones(self.n1, self.n1, device=self.device)
                                                - torch.diag(torch.ones(self.n1, device=self.device)))
        self.deletion_constant_matrix = 99999 * (torch.ones(self.n2, self.n2, device=self.device)
                                                - torch.diag(torch.ones(self.n2, device=self.device)))

        self.ot_scoring_layer = torch.nn.Linear(self.num_gcn_layers, 1)

        self.insertion_params, self.deletion_params = torch.nn.ParameterList([]), torch.nn.ParameterList([])
        for out_channels in filters:
            self.insertion_params.append(torch.nn.Parameter(torch.ones(out_channels)))
            self.deletion_params.append(torch.nn.Parameter(torch.zeros(out_channels)))

    def get_gnn_layer(self, in_channels, out_channels):
        if self.gnn_type == 'GCN':
            return pyg_nn.GCNConv(in_channels, out_channels)
        elif self.gnn_type == 'GIN':
            return pyg_nn.GINConv(torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(out_channels, out_channels)
            ))
        elif self.gnn_type == 'GraphSAGE':
            return pyg_nn.SAGEConv(in_channels, out_channels)
        elif self.gnn_type == 'GAT':
            return pyg_nn.GATConv(in_channels, out_channels)
        elif self.gnn_type == 'ChebNet':
            return pyg_nn.ChebConv(in_channels, out_channels, K=2)  # K is the filter size
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def GNN(self, data):
        """
        """
        gcn_feature_list = []
        features = data.x
        for conv_layer in self.conv_layers:
            features = conv_layer(features, data.edge_index)
            gcn_feature_list.append(features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features, p=self.dropout, training=self.training)
        return gcn_feature_list

    def forward(self, batch_data, batch_data_sizes, batch_adj):
        """
          batch_adj is unused
        """
        batch_sz = len(batch_data)
        q_graphs, c_graphs = zip(*batch_data)
        a, b = zip(*batch_data_sizes)
        qgraph_sizes = torch.tensor(a, device=self.device)
        cgraph_sizes = torch.tensor(b, device=self.device)
        query_batch = Batch.from_data_list(q_graphs)
        corpus_batch = Batch.from_data_list(c_graphs)
        query_gcn_feature_list = self.GNN(query_batch)
        corpus_gcn_feature_list = self.GNN(corpus_batch)

        pad_main_similarity_matrices_list = []
        pad_deletion_similarity_matrices_list = []
        pad_insertion_similarity_matrices_list = []
        pad_dummy_similarity_matrices_list = []

        for i in range(self.num_gcn_layers):
            q = pad_sequence(torch.split(query_gcn_feature_list[i], list(a), dim=0), batch_first=True)
            c = pad_sequence(torch.split(corpus_gcn_feature_list[i], list(b), dim=0), batch_first=True)
            q = F.pad(q, pad=(0, 0, 0, self.n1 - q.shape[1], 0, 0))
            c = F.pad(c, pad=(0, 0, 0, self.n2 - c.shape[1], 0, 0))
            # NOTE THE -VE HERE. BECAUSE THIS IS ACTUALLY COST MAT
            pad_main_similarity_matrices_list.append(-torch.matmul(q, c.permute(0, 2, 1)))

            pad_deletion_similarity_matrices_list.append(
                torch.diag_embed(-torch.matmul(q, self.deletion_params[i])) + self.insertion_constant_matrix)

            pad_insertion_similarity_matrices_list.append(
                torch.diag_embed(-torch.matmul(c, self.insertion_params[i])) + self.deletion_constant_matrix)

            pad_dummy_similarity_matrices_list.append(
                torch.zeros(batch_sz, self.n2, self.n1, dtype=q.dtype, device=self.device))

        sim_mat_all = []
        for j in range(batch_sz):
            for i in range(self.num_gcn_layers):
                a = pad_main_similarity_matrices_list[i][j]
                b = pad_deletion_similarity_matrices_list[i][j]
                c = pad_insertion_similarity_matrices_list[i][j]
                d = pad_dummy_similarity_matrices_list[i][j]
                s1 = qgraph_sizes[j]
                s2 = cgraph_sizes[j]
                sim_mat_all.append(torch.cat((torch.cat((a[:s1, :s2], b[:s1, :s1]), dim=1),
                                              torch.cat((c[:s2, :s2], d[:s2, :s1]), dim=1)), dim=0))

        sim_mat_all_cpu = [x.detach().cpu().numpy() for x in sim_mat_all]
        plans = [dense_wasserstein_distance_v3(x) for x in sim_mat_all_cpu]
        mcost = [torch.sum(torch.mul(x, torch.tensor(y, device=self.device, dtype=torch.float32))) for (x, y) in
                 zip(sim_mat_all, plans)]
        sz_sum = qgraph_sizes.repeat_interleave(self.num_gcn_layers) + cgraph_sizes.repeat_interleave(self.num_gcn_layers)
        mcost_norm = 2 * torch.div(torch.stack(mcost), sz_sum)
        scores_new = self.ot_scoring_layer(mcost_norm.view(-1, self.num_gcn_layers)).squeeze()

        if self.is_sig:
            return torch.sigmoid(scores_new).view(-1)
        else:
            return scores_new.view(-1)
