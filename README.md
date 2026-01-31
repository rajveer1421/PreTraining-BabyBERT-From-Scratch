# 🧠 Pretraining BabyBERT From Scratch

This repository is a **hands-on deep dive into the internal mechanics of BERT (Bidirectional Encoder Representations from Transformers)**.  
As part of the **IBM Generative AI: Language Modeling with Transformers** course, this project implements and **pretrains a compact “BabyBERT” model entirely from scratch using PyTorch**.

Rather than using a large 12-layer pretrained BERT, this project focuses on a **minimal, educational implementation** that preserves the **core pretraining objectives of BERT** while remaining computationally efficient and easy to understand.

---

## 📌 Project Motivation

BERT is often treated as a black box.  
This project breaks that abstraction by:

- Building **BERT-style inputs manually**
- Creating a **custom pretraining dataset**
- Training a **transformer encoder from random initialization**
- Implementing **MLM + NSP losses from scratch**

The goal is **conceptual clarity**, not leaderboard performance.

---

## 🧠 Key Learning Objectives

- Understand **bidirectional self-attention**
- Learn how **[CLS] and [SEP] tokens** are used
- Implement **Masked Language Modeling (MLM)**
- Implement **Next Sentence Prediction (NSP)**
- Train a transformer model **without pretrained weights**

---

## 📂 Dataset Creation (IMDB → BERT Pretraining CSV)

The IMDB dataset was refined into a **BERT-ready CSV dataset** with explicit supervision for both pretraining tasks.

### 📑 CSV Schema

| Column Name      | Description |
|------------------|-------------|
| `bert_input`     | Tokenized input sequence with randomly masked tokens |
| `bert_label`     | Original tokens (MLM targets) |
| `segment_label`  | Segment IDs (Sentence A / Sentence B) |
| `is_next`        | NSP label (1 = next sentence, 0 = random sentence) |

- **15% of tokens** are randomly masked
- Sentence pairs are generated for NSP
- Dataset is directly usable for BERT-style pretraining

---

## 📊 Kaggle Dataset

The complete preprocessed dataset is publicly available:

🔗 **Kaggle Dataset**  
https://www.kaggle.com/datasets/rajveergup1455/bert-dataset

This enables **reproducible BERT pretraining experiments**.

---

## 🏗️ BabyBERT Architecture

A lightweight transformer model that mirrors the **core structure of BERT**.

### 🔹 Configuration

- **Vocabulary Size:** 147,161
- **Embedding Dimension:** 10
- **Transformer Layers:** 2
- **Attention Heads:** Multi-head self-attention
- **Training Objectives:** MLM + NSP
- **Parallelism:** `torch.nn.DataParallel`

---

### 🔍 Full Model Architecture

```text
DataParallel(
  (module): BERT(
    (embedding): BERTEmbedding(
      (token_embedding): TokenEmbedding(
        (embedding): Embedding(147161, 10)
      )
      (positional_encoding): PositionalEncoding(
        (dropout): Dropout(p=0.1)
      )
      (segment_embedding): Embedding(3, 10)
      (dropout): Dropout(p=0.1)
    )

    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0-1): 2 x TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): Linear(10 → 10)
          )
          (linear1): Linear(10 → 2048)
          (linear2): Linear(2048 → 10)
          (norm1): LayerNorm(10)
          (norm2): LayerNorm(10)
          (dropout): Dropout(p=0.1)
        )
      )
    )

    (encoder_norm): LayerNorm(10)

    (nsp_head): Sequential(
      Linear(10 → 10)
      GELU
      Dropout(p=0.1)
      Linear(10 → 2)
    )

    (mlm_transform): Sequential(
      Linear(10 → 10)
      GELU
      LayerNorm(10)
    )

    (mlm_decoder): Linear(10 → 147161)

    (dropout): Dropout(p=0.1)
  )
)
## 🚀 Training Details

| Setting | Value |
|--------|------|
| Dataset | IMDB-based BERT CSV (Kaggle) |
| Training Type | Pretraining from scratch |
| Objectives | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |
| Epochs | 100 |
| Optimizer | Adam |
| Framework | PyTorch |
| Parallel Training | `torch.nn.DataParallel` |

The model is initialized with **random weights** and learns contextual representations purely through **self-supervised learning objectives**.

---

## 🧪 Pretraining Tasks Explained

### 🔹 Masked Language Modeling (MLM)

- **15% of input tokens** are randomly masked  
- The model predicts the original tokens using **both left and right context**  
- Enables **true bidirectional language understanding**

---

### 🔹 Next Sentence Prediction (NSP)

- The model predicts whether **Sentence B logically follows Sentence A**
- Uses the **`[CLS]` token representation** for binary classification
- Helps the model learn **sentence-level coherence and relationships**

---

## 🛠️ Tech Stack

- **Framework:** PyTorch   
- **Data Handling:** `torchtext`, `torch.utils.data.DataLoader`  
- **Parallelism:** `torch.nn.DataParallel`  
- **Course Context:** IBM Generative AI Specialization
