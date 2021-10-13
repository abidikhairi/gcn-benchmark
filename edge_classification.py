import wandb
import torch
import dgl

import torchmetrics.functional as metrics
from dgl.data import WN18Dataset

from models.models import EdgeClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    dataset = WN18Dataset(reverse=False, verbose=True)

    graph = dataset[0]

    train_mask = graph.edata['train_mask']
    test_mask = graph.edata['test_mask']

    train_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
    test_idx = torch.nonzero(test_mask, as_tuple=False).flatten()

    graph = dgl.add_self_loop(graph)

    return graph, train_idx, test_idx

def main():
    experiment = wandb.init(project='GCN', entity='flursky', name='GCN Link Classification WN18 Dataset')


    graph, train_idx, test_idx = load_data()
    graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1)
    
    num_classes = len(torch.unique(graph.edata['etype']))
    
    print('# edges for training:', 141442)
    print('# edges for testing:', 5000)
    print('# classes:', num_classes)

    model = EdgeClassifier(feature_size=1, hidden_size=64, mlp_hidden_size=64, num_classes=num_classes, dropout_rate=0.5)
    criterion = torch.nn.NLLLoss()
    adam = torch.optim.Adam(model.parameters(), lr=1e-2)

    wandb.watch(model)

    model.to(device)
    graph = graph.to(device)

    for _ in range(20):
        model.train()
        adam.zero_grad()

        n_feats = graph.ndata['feat'].to(device)
        labels = graph.edata['etype'].to(device)

        logits = model(graph, n_feats)

        loss = criterion(logits[train_idx], labels[train_idx])

        loss.backward()
        adam.step()

        wandb.log({
            'training_loss': loss
        })

        with torch.no_grad():
            model.eval()

            n_feats = graph.ndata['feat'].to(device)
            labels = graph.edata['etype'].to(device)

            logits = model(graph, n_feats)
            loss = criterion(logits[test_idx], labels[test_idx])

            accuracy = metrics.accuracy(logits, labels)

            wandb.log({
                'train_loss': loss,
                'accuracy': accuracy
            })
    
    experiment.finish()

if __name__ == '__main__':
    main()
