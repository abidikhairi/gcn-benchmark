import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
import matplotlib.pyplot as plt
from dgl.data import CoraGraphDataset

from models import GCN, GCNWithBlocks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    dataset = CoraGraphDataset(verbose=False)
    graph = dataset[0].to(device)
    
    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']

    criterion = torch.nn.NLLLoss()

    gcn = GCN(feature_size=1433, hidden_size=16, num_classes=7)
    blocks = GCNWithBlocks(feature_size=1433, hidden_size=16, num_classes=7)
    
    gcn.to(device)
    blocks.to(device)

    optimizerA = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizerB = torch.optim.Adam(blocks.parameters(), lr=1e-3, weight_decay=5e-4)
    
    data = {
        'blocks': {
            'test_loss': [],
            'accuracy': []
        },
        'gcn':  {
            'test_loss': [],
            'accuracy': []
        },
    }

    for _ in range(100):
        gcn =  train(gcn, optimizerA, criterion, graph, train_mask)
        blocks = train(blocks, optimizerB, criterion, graph, train_mask)

        acc_blocks, test_loss_blocks = evaluate(gcn, graph, criterion, test_mask)
        acc_gcn, test_loss_gcn = evaluate(blocks, graph, criterion, test_mask)

        data['blocks']['test_loss'].append(test_loss_blocks.item())
        data['blocks']['accuracy'].append(acc_blocks.item())

        data['gcn']['test_loss'].append(test_loss_gcn.item())
        data['gcn']['accuracy'].append(acc_gcn.item())

    x = torch.arange(100).numpy()

    plt.figure(figsize=(10, 8))
    plt.title('Flat GCN Vs Block GCN (test loss)')
    plt.plot(x, data['gcn']['test_loss'], label='gcn')
    plt.plot(x, data['blocks']['test_loss'], label='blocks')
    plt.legend(loc='upper left')
    plt.savefig('images/flat_vs_block_test_loss.png')

    plt.figure(figsize=(10, 8))
    plt.title('Flat GCN Vs Block GCN (accuracy)')
    plt.plot(x, data['gcn']['accuracy'], label='gcn')
    plt.plot(x, data['blocks']['accuracy'], label='blocks')
    plt.legend(loc='upper left')
    plt.savefig('images/flat_vs_block_acc.png')


def train(model, optimizer, criterion, graph, mask):
    model.train()

    optimizer.zero_grad()
    n_feats = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    out = model(graph, n_feats)
    logits = F.log_softmax(out, dim=1)

    loss = criterion(logits[mask], labels[mask])

    loss.backward()
    optimizer.step()

    return model

def evaluate(model, graph, criterion, mask):
    model.eval()
    with torch.no_grad():
        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = model(graph, n_feats)
        logits = F.log_softmax(out, dim=1)

        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        test_loss = criterion(logits, labels)
        acc = metrics.accuracy(logits[mask], labels[mask])

        return acc, test_loss


if __name__ == '__main__':
    main()
