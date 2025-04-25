![image](https://github.com/user-attachments/assets/eb1baddd-4903-431e-b352-44d5d57784d2)

Implementation for the paper [Speculative RAG](https://arxiv.org/abs/2407.08223) (ICLR 2025)

## Installation
You can directly create a conda environment using the provided configuration file.
```bash
cd Speculative_RAG
conda env create -f environment.yml
```

## Evaluation
Using the following script to evaluate Speculative RAG.
```bash
conda activate sperag
cd Speculative_RAG
bash ./scripts/inference.sh
```
