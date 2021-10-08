# Machine Learning Project Template

This is a project template to help me go faster when testing new models



### Project Requirements

- torch
- dgl
- wandb
- matplotlib
- numpy


## Node Classification Test

#### Cora Dataset
- Nodes: 2708
- Edges: 10556
- Graph Diameter: 20
- Node Features: 1433
- Num Classes: 7

<img src="./images/cora_train_loss.png" width="400"/>  <img src="./images/cora_test_loss.png" width="400"/> <img src="./images/cora_accuracy.png" width="400"/>


#### Pubmed Dataset
- Nodes: 19717 
- Edges: 88651
- Graph Diameter: 3.7 ~ 4 (90-percentile effective diameter)
- Node Features: 500
- Num Classes: 3

<img src="./images/pubmed_train_loss.png" width="400"/>  <img src="./images/pubmed_test_loss.png" width="400"/> <img src="./images/pubmed_accuracy.png" width="400"/>

#### What each node in GNN see with K layers

##### 2 Layer GCN
![2 layer GNN](images/gcn-2-layers.png)

##### 10 Layer GCN
![10 layer GNN](images/gcn-10-layers.png)


#### Over Smoothing problem in GNNs

![depth_diameter](images/depth_vs_diameter.png)

#### Feature transformation in GCN

![features_tsne](images/cora_features.png)
