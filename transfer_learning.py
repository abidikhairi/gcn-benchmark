import wandb
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

from models import GCNTransferLearning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cora():
    dataset = CoraGraphDataset(verbose=False)

    graph = dataset[0].to(device)

    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']

    return graph, train_mask, test_mask

def load_citeseer():
    dataset = CiteseerGraphDataset(verbose=False)
    graph = dataset[0].to(device)

    test_mask = graph.ndata['test_mask']
    train_mask = graph.ndata['train_mask']

    return graph, train_mask, test_mask

def main():
    experiment = wandb.init(project='GCN', entity='flursky', name='GCN Transfer Learning', reinit=True)

    gcn = GCNTransferLearning(feature_size=1433, hidden_size=16, num_classes=7, dropout_rate=0.5)
    criterion = torch.nn.NLLLoss()
    optimizerA = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=5e-4)
    
    model = train(gcn, optimizerA, criterion, 'cora')
    
    # Freeze GCN parameters
    model.feature.requires_grad_(False)
    
    # Adapt Classifer to new Dataset
    model.projection = torch.nn.Sequential(
        torch.nn.Linear(in_features=3703, out_features=16),
        torch.nn.Dropout(0.5)
    )

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=16, out_features=6)
    )

    optimizerB = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    train(model, optimizerB, criterion, 'citeseer')

    experiment.finish()

def train(model, optimizer, criterion, dataset):
    if dataset == 'cora': graph, train_mask, test_mask = load_cora()
    elif dataset == 'citeseer': graph, train_mask, test_mask = load_citeseer()
    else: raise ValueError('dataset not supported')

    model.to(device)

    for _ in range(1000):
        model.train()

        optimizer.zero_grad()
        n_feats = graph.ndata['feat'].to(device)
        labels = graph.ndata['label'].to(device)

        out = model(graph, n_feats)

        logits = F.log_softmax(out, dim=1)

        loss = criterion(logits[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()

        wandb.log({'{}_train_loss'.format(dataset): loss})

        with torch.no_grad():
            model.eval()

            out = model(graph, n_feats)

            logits = F.log_softmax(out, dim=1)

            labels = labels.detach().cpu()
            logits = logits.detach().cpu()


            test_loss = criterion(logits[test_mask], labels[test_mask])
            acc = metrics.accuracy(logits[test_mask], labels[test_mask])

            wandb.log({
                '{}_test_loss'.format(dataset): test_loss,
                '{}_accuracy'.format(dataset): acc
            })
    return model

if __name__ == '__main__':
    main()
