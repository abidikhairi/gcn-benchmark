import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
import matplotlib.pyplot as plt
from dgl.data import CoraGraphDataset

from models import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    dataset = CoraGraphDataset(verbose=False)
    graph = dataset[0].to(device)

    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    cirterion = torch.nn.NLLLoss()
    
    accuracies = []
    widths = [2 ** i for i in range(1, 11)]

    
    for hidden_size in widths:
    
        model = GCN(feature_size=1433, hidden_size=hidden_size, num_classes=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        
        model = train(model, optimizer, cirterion, graph, train_mask)

        acc = evaluate(model, graph, test_mask)

        accuracies.append(acc)


    x = widths
    y = torch.tensor(accuracies).numpy()
    
    plt.title('GCN Width Vs Accuracy')
    plt.plot(x, y)
    plt.xlabel('width (hidden_size)')
    plt.xticks(x)
    plt.xscale('linear')
    plt.ylabel('accuracy')
    plt.savefig('images/gcn_width.png')

def train(gcn ,optimizer, cirterion, graph, mask):
    gcn.to(device)
    gcn.train()
    for _ in range(100):
        optimizer.zero_grad()

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = gcn(graph, n_feats)

        logits = F.log_softmax(out, dim=1)

        loss = cirterion(logits[mask], labels[mask])

        loss.backward()
        optimizer.step()
    
    return gcn

def evaluate(model, graph, mask):
    model.eval()
    with torch.no_grad():
        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = model(graph, n_feats)
        logits = F.log_softmax(out, dim=1)

        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        acc = metrics.accuracy(logits[mask], labels[mask])

        return acc

if __name__ == '__main__':
    main()