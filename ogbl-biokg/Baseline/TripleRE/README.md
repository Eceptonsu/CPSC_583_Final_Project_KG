# Knowledge Graph Embeddings: TripleRE

TThe code is comes from an open source code repo at [https://github.com/yulong-CSAI/TripleRE-biokg](https://github.com/yulong-CSAI/TripleRE-biokg).

Due to Python multithreading challenges, this codebase may encounter issues when run locally. To address this, we have developed a Google Colab notebook, `TripleRE.ipynb`, which enables execution on a cloud server.

## Training the Model

To train the model, please access the Google Colab notebook. Modify the directory paths as needed for your setup.

For local execution, you can use the command:

```bash
python run.py --do_train --cuda --do_valid --do_test --evaluate_train --model TripleRE -n 128 -b 512 -d 5000 -g 12 -a 1.0 -adv -tr -lr 0.0005 --max_steps 100000 --cpu_num 4 --test_batch_size 32 --save_path ./
```

## Rank File Generation

Currently, the code in this folder does not support rank file generation. Please refer to the [Others](https://github.com/Eceptonsu/CPSC_583_Final_Project_KG/tree/main/ogbl-biokg/Baseline/Others) folder for more info.