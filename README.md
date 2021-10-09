# Graph Convolution Networks Benchmarks

In This project i tested gcn in a bunch of ML Tasks
- Node Classification
- The impact of GCN Depth/Width on accuracy
- Apply Transfer Learning on Node Classification
- Visualize K-hops neighbors in Graphs


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

#### GCN Width Vs Accuracy

![width_accuracy](images/gcn_width.png)


#### Feature transformation in GCN

![features_tsne](images/cora_features.png)

### Block Based GCN Vs Flat GCN

##### GNN Blocks Design
![gnn_blocks](images/GNNBlocks.png) 

![flat_vs_bloc_acc](images/flat_vs_block_acc.png) ![flat_vs_bloc_test_loss](images/flat_vs_block_test_loss.png)



### References

```
@misc{kipf2017semisupervised,
      title={Semi-Supervised Classification with Graph Convolutional Networks}, 
      author={Thomas N. Kipf and Max Welling},
      year={2017},
      eprint={1609.02907},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{wandb,
title = {Experiment Tracking with Weights and Biases},
year = {2020},
note = {Software available from wandb.com},
url={https://www.wandb.com/},
author = {Biewald, Lukas},
}

@misc{wang2020deep,
      title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks}, 
      author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
      year={2020},
      eprint={1909.01315},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```