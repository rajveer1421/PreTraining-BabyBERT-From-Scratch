# PreTraining BabyBERT From Scratch 👶🚀

This repository is a deep-dive exploration into the internal mechanics of the **BERT (Bidirectional Encoder Representations from Transformers)** architecture. Developed as part of the **IBM Generative AI: Language Modeling with Transformers** course, this project focuses on building and pre-training a "BabyBERT" model from the ground up using PyTorch.

## 📌 Project Overview
The goal of this project is to demonstrate how BERT learns to understand language context through self-supervised learning. Instead of using a massive 12-layer model, we implement a smaller version (**BabyBERT**) to keep training efficient while retaining the core logic of the original paper.

### Key Learning Objectives:
* **Data Preprocessing:** Implementing custom data pipelines for BERT-style inputs.
* **Tokenization:** Utilizing `transformers.BertTokenizer` for subword encoding.
* **Architectural Mastery:** Understanding the role of the `[CLS]` and `[SEP]` tokens.
* **Pre-training Tasks:** Implementing **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.

---

## 🏗️ Architecture & Core Concepts

### 1. The [CLS] Token
In this implementation, we use the `[CLS]` (Classification) token as a global summary of the input sequence. After passing through the Transformer layers, the final hidden state of this token is used for the **NSP task**.

### 2. Masked Language Modeling (MLM)
Masking is the secret sauce that makes BERT bidirectional. 
* **How it works:** We randomly mask 15% of the input tokens.
* **The Goal:** The model must predict the original tokens based only on the context provided by non-masked words.
* **Why Masking?** Unlike standard models that read left-to-right, masking forces BERT to look at both the left and right context simultaneously to "fill in the blanks."



### 3. Next Sentence Prediction (NSP)
To understand the relationship between different sentences, the model is trained to predict if Sentence B logically follows Sentence A. We use the `[SEP]` token to delineate sentence boundaries.

---

## 🛠️ Tech Stack
* **Framework:** [PyTorch](https://pytorch.org/)
* **Tokenization:** `transformers.BertTokenizer` (Hugging Face)
* **Data Handling:** `torchtext` & `torch.utils.data.DataLoader`
* **Course Context:** Part of the *IBM Generative AI Specialization*.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers torchtext