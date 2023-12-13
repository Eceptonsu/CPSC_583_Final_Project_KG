# CPSC_583_Final_Project_KG: MovieLens100K Dataset

## Preliminary Data Analysis
To generate the preliminary data analysis results and corresponding plots on the MovieLens100K dataset, please run the notebook: Movielens_PreliminaryAnalysis.ipynb

## Traditional Link Prediction
To generate the traditional link prediction models' performance results on the MovieLens100K dataset, please run the notebook: Movielens_LinkPrediction.ipynb\
In the class "Embedder", you can change the variable self.gnn to one of the 6 following model variables: "self.sage, self.sage_res, self.sage_res_2, self.gat, self.gat_res, self.gat_res_2" (e.g. self.gnn = self.gat_res_2) to run different models
To generate the traditional link prediction models' performance results with different loss functions on the MovieLens100K dataset, please run the notebook: Movielens_LinkPrediction_Different_Loss_Functions.ipynb\
In the function train, you can change the value of mode to test different loss functions, you can also add your own loss functions to the get_loss function for testing.
