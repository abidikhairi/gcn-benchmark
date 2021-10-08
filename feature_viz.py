import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dgl.data import CoraGraphDataset
from sklearn.manifold import TSNE

from models import GCN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    dataset = CoraGraphDataset(verbose=False)
    graph = dataset[0].to(device)

    model = GCN(feature_size=1433, hidden_size=16, num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    model.to(device)

    for _ in range(100):

        optimizer.zero_grad()

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = model(graph, n_feats)

        logits = F.log_softmax(out, dim=1)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
    
    n_feats = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].cpu().numpy()

    tsne = TSNE(n_components=2)

    preds = model(graph, n_feats)

    features = tsne.fit_transform(preds.detach().cpu().numpy())
    
    plt.figure(figsize=(14, 12))
    plt.title('2D Cora Features Visualization')
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='Dark2')
    plt.savefig('images/cora_features.png')

if __name__ == '__main__':
    main()
