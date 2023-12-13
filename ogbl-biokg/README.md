# CPSC_583_Final_Project_KG: BioKG Dataset

## Preliminary Data Analysis
To generate the preliminary data analysis results and corresponding plots on the BioKG dataset, please run the notebook: [BioKG_PreliminaryAnalysis](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/blob/main/ogbl-biokg/BioKG_LinkPrediction.ipynb)

## Traditional Link Prediction
Similar experiments were attempted on the BioKG dataset as well. However, due to the complexity of the KG and limitation of traditional link prediction methods on heterogeneous graphs, the current framework could not handle the BioKG dataset’s structure well. Therefore, more advanced methods to handle massive KGs are required beyond the traditional GNN scope. To view the existing code for traditional link prediction on BioKG, please run the notebook [BioKG_PreliminaryAnalysis](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/blob/main/ogbl-biokg/BioKG_PreliminaryAnalysis.ipynb)

## Knowledge Graph Embedding and Ensemble Learning

### Baseline Model Training
To perform ensemble learning on KG embeddings, we need to first train baseline models. For more information, please refer to instructions in [Baseline](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/tree/main/ogbl-biokg/Baseline) folder.

### Rank File Generation
After we have trained baseline models, we can generate validation and test rank files for each of the model. This step is *crucial* because these files store the outputs of individual Knowledge Graph Embedding (KGE) models. For more information, please refer to instructions in [Baseline](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/tree/main/ogbl-biokg/Baseline) folder.

NOTE: For OGB datasets, the shape of the ranks npy file is (#number of samples, 1002). It includes 1002 columns, which consist of 1 positive head rank, 500 head negative ranks, 1 positive tail rank, and 500 tail negative ranks.

### Preparation for Ensemble Learning
The directory should be organized as follows:
```bash
|-- CPSC_583_Dataset
    |-- {dataset1_name}
        |-- {model1_name}_valid_ranks.npy
        |-- {model1_name}_test_ranks.npy
    |-- {dataset2_name}
        |-- {model2_name}_valid_ranks.npy
        |-- {model2_name}_test_ranks.npy
```

### Actual Ensemble Learning
To perform the ensemble learning, please run the notebook: [Knowledge_Graph_Embeddings](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/blob/main/ogbl-biokg/Knowledge_Graph_Embeddings.ipynb)

To enable ensemble learning with dynamic weight adjustment, set `dynamic_weight_adjust` to `True`. You can also change other configurations.

## Genetic Algorithm
To perform the genetic algorithm, please run the notebook: [Knowledge_Graph_Embeddings](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/blob/main/ogbl-biokg/Knowledge_Graph_Embeddings.ipynb) and refer to the **Genetic Algorithm** Section.
