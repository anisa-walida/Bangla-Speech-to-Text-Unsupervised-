# Bangla-Speech-to-Text-Unsupervised-
This repository contains my implementation for fine-tuning a **Wav2Vec2-CTC** model on **Bengali** speech data using the [Hugging Face Transformers](https://huggingface.co/transformers/) and `datasets` libraries. The training was done efficiently on **Kaggle Notebooks**, carefully managing memory and disk constraints while maintaining training continuity across multiple data chunks.

## 📌 Overview

Automatic Speech Recognition (ASR) for low-resource languages like Bengali is a challenging task due to the scarcity of labeled data. To address this, I fine-tuned a pretrained **Wav2Vec2** model on a preprocessed Bengali speech dataset. The training pipeline supports **chunked training**, **resume-from-checkpoint**, and **automatic model state management** (optimizer, scheduler, tokenizer).

---

## 🔍 Features

- ✅ Pretraining Hugging Face's `facebook/wav2vec2-base` 
- ✅ Fine-tuning Hugging Face's `facebook/wav2vec2-base` with **Connectionist Temporal Classification (CTC) loss**
- ✅ Training on **preprocessed `.parquet` chunks** for large-scale audio data
- ✅ Using the pretrained and finetuned wav2vec model in Bangla Asr pipeline
- ✅ Custom data collator for preprocessed inputs
- ✅ Uploading and syncing final model to Hugging Face Hub

---


| Area | Toola |
|------|-------|
| **Machine Learning** | Transfer learning, CTC loss, optimizer scheduling |
| **NLP / Speech** | ASR modeling, feature extraction, tokenization |
| **Tooling** | Hugging Face Transformers, Datasets, PyTorch |
| **Engineering** | Chunk-based training, resume-safe checkpoints, memory-safe pipelines |
| **MLOps** | Model versioning, remote uploads, automated tracking |

---
