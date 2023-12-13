# CPSC_583_Final_Project_KG
This is the github repository for CPSC 583 Final Project by Bowen Duanmu, Evan Shi, Andrew Yi

## Project Overview
Our project can be devided into three main part:

1. We explored the traditional GNN methods' effectiveness and limits on tackling link prediction tasks on complex heterogeneous Knowledge Graphs.  
2. We utilized more cutting-edge methods such as KG Embeddings and Ensemble learning to attempt the same link prediction tasks. 
3. We also proposed further improvements such as utilizing Genetic Algorithm.

## Datasets
To generate the traditional link prediction models' performance results on the MovieLens100K dataset, please run the notebook: Movielens_LinkPrediction.ipynb\
In the class "Embedder", you can change the variable self.gnn to one of the 6 following model variables: "self.sage, self.sage_res, self.sage_res_2, self.gat, self.gat_res, self.gat_res_2" (e.g. self.gnn = self.gat_res_2) to run different models
