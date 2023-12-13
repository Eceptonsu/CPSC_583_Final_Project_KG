# CPSC_583_Final_Project_KG
This is the github repository for Fall 23 CPSC 583 Final Project by Bowen Duanmu, Evan Shi, Andrew Yi.

## Project Overview
Our project can be devided into three main part:

1. Traditional GNN methods' effectiveness and limits on tackling link prediction tasks on complex heterogeneous KGs.  
2. More cutting-edge methods such as KG Embeddings and ensemble learning to attempt the same link prediction tasks. 
3. Genetic Algorithm.

## Dependencies
Please first select a version and install torch with CUDA

Then install pytorch geometric

```bash
pip install torch_geometric
```

Make sure the installation is performed correctly

```bash
python -c "import torch; print(torch.version.cuda)"
```

And then install other pyg/torch related dependencies

```basg 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Where `TORCH` and `CUDA` are replaced by the specific versions according to the official documentation https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Then, you can simply do

```bash
pip install -r requirements.txt
```

For the rest of the dependencies

## Running the Project
Please refer to the individual README.md files in the [MovieLens](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/tree/main/MovieLens) and [obgl-biokg](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/tree/main/ogbl-biokg) directories for separate instructions

