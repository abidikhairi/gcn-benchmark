from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, feature_size, hidden_size, num_classes, dropout_rate = 0.5):
        super(GCN, self).__init__()

        
        self.gcn = nn.Sequential(
            GraphConv(in_feats=feature_size, out_feats=hidden_size, activation=nn.ReLU()),
            nn.Dropout(dropout_rate),
            GraphConv(in_feats=hidden_size, out_feats=num_classes),
        )

    def forward(self, graph, n_feats):
        x = n_feats

        for layer in self.gcn:
            if isinstance(layer, GraphConv):
                x = layer(graph, x)
            else:
                x = layer(x)
        
        return x