import sys
import wandb
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from dgl.data import CoraGraphDataset, PubmedGraphDataset
from models import GCN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cora_benchmark():
    experiment = wandb.init(project='GCN', entity='flursky', name='GCN Cora Benchmark', reinit=True)

    dataset = CoraGraphDataset(verbose=False)
    graph = dataset[0].to(device)

    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']


    gcn = GCN(feature_size=1433, hidden_size=16, num_classes=7)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-2, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    wandb.watch(gcn)
    gcn.to(device)

    for _ in range(100):
        gcn.train()
        optimizer.zero_grad()

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = gcn(graph, n_feats)
        logits = F.log_softmax(out, dim=1)


        loss = criterion(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()

        wandb.log({
            'train_loss': loss
        })

        evaluate(gcn, graph, criterion, test_mask)

    experiment.finish()

def pubmed_benchmark():
    experiment = wandb.init(project='GCN', entity='flursky', name='GCN Pubmed Benchmark', reinit=True)

    dataset = PubmedGraphDataset(verbose=False)
    graph = dataset[0].to(device)

    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']


    gcn = GCN(feature_size=500, hidden_size=16, num_classes=3)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-2, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    wandb.watch(gcn)
    gcn.to(device)

    for _ in range(100):
        gcn.train()
        optimizer.zero_grad()

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = gcn(graph, n_feats)
        logits = F.log_softmax(out, dim=1)


        loss = criterion(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()

        wandb.log({
            'train_loss': loss
        })

        evaluate(gcn, graph, criterion, test_mask)

    experiment.finish()

def evaluate(model, graph, criterion, mask):
    model.eval()
    with torch.no_grad():

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = model(graph, n_feats)
        logits = F.log_softmax(out, dim=1)

        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        test_loss = criterion(logits[mask], labels[mask])
        acc = metrics.accuracy(logits, labels)

        wandb.log({
            'accuracy': acc,
            'test_loss': test_loss 
        })


if __name__ == '__main__':
    try:
        dataset = sys.argv[1]
        if dataset == 'cora': cora_benchmark()
        if dataset == 'pubmed': pubmed_benchmark() 
    except IndexError:
        raise ValueError('no dataset provided expected one of: pubmed, cora')