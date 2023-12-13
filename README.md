# CPSC_583_Final_Project_KG
This is the github repository for CPSC 583 Final Project by Bowen Duanmu, Evan Shi, Andrew Yi

## Project Overview
Our project can be devided into three main part:

1. We explored the traditional GNN methods' effectiveness and limits on tackling link prediction tasks on complex heterogeneous Knowledge Graphs.  
2. We utilized more cutting-edge methods such as KG Embeddings and Ensemble learning to attempt the same link prediction tasks. 
3. We also proposed further improvements such as utilizing Genetic Algorithm.

## Dependencies
Please first select a version and install torch with CUDA\
Then do\
`pip install torch_geometric`\
`python -c "import torch; print(torch.version.cuda)"`\
And then\
`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`\
Where TORCH and CUDA are replaced by the specific torch and CUDA versions\
(according to the official documentation https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)\

Then, you can simply do\
`pip install -r requirements.txt`\
For the rest of the dependencies

## Running the Project
Please refer to the individual README.md files in the MovieLens and obgl-biokg directories for separate instructions

