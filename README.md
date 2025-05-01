![image](https://github.com/user-attachments/assets/eb1baddd-4903-431e-b352-44d5d57784d2)

Pytorch Implementation for the paper [Speculative RAG](https://arxiv.org/abs/2407.08223) (ICLR 2025)

## Installation
You can directly create a conda environment using the provided configuration file.
```bash
cd Speculative_RAG
conda env create -f environment.yml
```

## Embedding & Retrieval
```bash
conda activate sperag
cd Speculative_RAG
bash ./scripts/embedding.sh
```

## Evaluation
Using the following script to evaluate Speculative RAG.
```bash
conda activate sperag
cd Speculative_RAG
bash ./scripts/inference.sh
```

## Fine-tuning (Option)
To fine-tune the RAG-drafter, first activate the environment and then run the script below. Be sure to update the input_file, model, and output_dir based on your setup.
```bash
conda activate sperag
cd Speculative_RAG
bash ./scripts/finetuning.sh
```
