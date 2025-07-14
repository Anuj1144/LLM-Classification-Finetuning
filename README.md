# LLM-Classification-Finetuning
Finetune LLMs to Predict Human Preference using Chatbot Arena conversations



# 🧠 LLM Preference Classifier

A transformer-based model fine-tuned to predict human preferences between the responses of two Large Language Models (LLMs), using prompts and their generated outputs.

## 🚀 Project Overview

This project tackles the **LLM Comparison Problem**, where the task is to choose the better response between two candidate answers (`response_a` and `response_b`) to a given `prompt`. The model predicts whether:

- `response_a` is preferred (label: 0)
- `response_b` is preferred (label: 1)
- or the result is a tie (label: 2)

---

## 📂 Dataset

The dataset includes:
- `id`: Unique sample ID
- `prompt`: The input prompt
- `response_a`, `response_b`: Generated responses by different LLMs
- `winner_model_a`, `winner_model_b`, `winner_tie`: Ground truth labels in one-hot format

### Data Source
- Uploaded via [Kaggle Datasets](https://www.kaggle.com/)

---

## 🧠 Model

We fine-tuned `microsoft/deberta-v3-small` on this custom classification task.

### Why DeBERTa?
- Superior performance in understanding long-form, complex natural language.
- Lightweight enough for fast training with high accuracy.

### Training Details

- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Scheduler: CosineLR with warmup
- Batch size: 8 (train), 32 (eval)
- Epochs: 20
- Dropout: 0.5
- Early stopping: Enabled (patience = 2)

---

## 📊 Evaluation

| Metric     | Score     |
|------------|-----------|
| Accuracy   | ~46%      |
| F1-score   | ~46%      |
| Precision  | ~47%      |
| Recall     | ~46%      |

These scores show that the model learns patterns of preference with moderate success, though further fine-tuning and dataset balancing could improve performance.

---

## 📦 Files

- `train.csv`, `test.csv`: Labeled and unlabeled data
- `submission.csv`: Output prediction file (Kaggle format)
- `notebook.ipynb`: Kaggle notebook containing full training + inference pipeline
- `results/`: Final saved model weights and logs

---

## 🛠 Technologies Used

- Python 3
- Hugging Face Transformers
- PyTorch
- scikit-learn
- Datasets (HuggingFace)
- Kaggle Kernels (GPU runtime)

---

## 📝 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/llm-preference-classifier.git
   cd llm-preference-classifier

