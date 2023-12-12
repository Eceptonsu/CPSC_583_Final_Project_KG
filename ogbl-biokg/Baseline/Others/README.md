# Knowledge Graph Embeddings: Rescal, CP, TransE, RotateE, TuckER, ComplEx-RP, UniBi

This repository contains modified code originally sourced from an open-source repository at [https://github.com/facebookresearch/ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction). Our modifications enable:
1. Application of the code to models beyond ComplEX-RP.
2. Generation of rank files from a pretrained model.

## Training the Model
Before training, ensure the `preprocess_datasets.py` script is configured with the correct datasets:

```bash
# Inside preprocess_datasets.py
datasets = ['ogbl-biokg']
```

Run the preprocessing script:

```bash
mkdir data/
python preprocess_datasets.py
```

After preprocessing, train a model by executing `main.py`. For instance, to train a ComplEx model on ogbl-biokg, use:

```bash
python main.py --dataset ogbl-biokg --model ComplEx --score_rel True --rank 1000 --learning_rate 1e-1 --batch_size 500 --optimizer Adagrad --regularizer N3 --lmbda 0.01 --w_rel 0.25 --valid 1 --model_cache_path ./
```

Adjust the command arguments as necessary for different configurations.

## Rank File Generation

Generate rank files for various models using the `--generate_rank` argument, and specify the model path with `--model_path`. For example:

```bash
python main.py --dataset ogbl-biokg --model RotatE --rank 1000 --generate_rank True --model_path ./model_path_here
```