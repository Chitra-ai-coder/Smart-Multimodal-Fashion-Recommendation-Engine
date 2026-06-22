# 🛍️ Smart Multimodal Fashion Recommendation Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

**Author:** Chitrabhanu Hazra  
**Platform:** Kaggle  
**Dataset:** Myntra Fashion Product Images (Small)

---

## 📖 Project Overview
This project implements an autonomous, multimodal artificial intelligence catalog for e-commerce fashion. It moves beyond traditional, metadata-reliant search engines by combining high-dimensional visual mathematics with natural language processing. 

The system acts as an intelligent shopping assistant that can visually deduplicate inventory, perform natural language reverse-image searches, and logically recommend culturally and contextually relevant clothing accessories without requiring manual human tagging.

---

## ⚙️ Architecture & Technology Stack

### Core AI Models
* **Visual Encoder (OpenAI CLIP):** Utilized a ViT (Vision Transformer) architecture to generate 512-dimensional mathematical embeddings of product images, projecting text and images into a shared latent space.
* **Logic Engine (Google Flan-T5-Base):** An instruction-finetuned sequence-to-sequence Large Language Model (LLM) serving as the "fashion consultant" to deduce logical clothing combinations.

### Frameworks & Libraries
* **Hugging Face `transformers`:** Managed model pipelines, tokenization, and image processing.
* **PyTorch:** Provided the deep learning tensor backend and hardware acceleration for matrix multiplication.
* **Scikit-Learn:** Executed Cosine Similarity calculations for n-dimensional vector matching.
* **Matplotlib & PIL:** Handled dynamic image rendering and UI visualization.

---

## 🚀 Core Features

### 1. Mathematical Data Deduplication (Catalog Cleaning)
Real-world datasets contain heavy duplication. This module passes all inventory through the CLIP Image Encoder. It calculates a similarity matrix using cosine similarity and mathematically purges duplicate vendor images (similarity threshold `> 0.98`), resulting in a pristine, unique `clean_catalog_df`.

### 2. Reverse Product Search
Allows users to query the visual database using natural language (e.g., *"blue casual shirt"*). The text query is encoded via the CLIP Text Encoder and mathematically matched against the visual vectors of the catalog, bypassing the need for manual image tags entirely.

### 3. Smart Recommendation Engine
A highly integrated pipeline combining logical reasoning with visual retrieval:
1. **The Brain:** The user inputs an item (e.g., *"Running Shoes"*). Flan-T5 logically deduces 5 complementary accessories (e.g., *"socks, shorts, headband, watch, towel"*).
2. **The Eyes:** The text suggestions are passed into the CLIP visual engine to retrieve the exact matching products from the unique Myntra catalog.
3. **The Display:** Results are rendered in a clean, dynamic Matplotlib grid.

---

## 🛠️ Engineering Challenges & Solutions

During development, several complex machine learning challenges were addressed to ensure production-grade stability:

* **Multimodal Projection Alignment:** Resolved a known library versioning bug where specific wrapper configurations skipped the visual projection layer. Fixed by injecting a blank dummy image tensor `Image.new('RGB', (224, 224), (0, 0, 0))` alongside text queries to force correct multimodal alignment during inference.
* **Mitigating LLM Hallucinations:** Base-level LLMs are prone to formatting loops and prompt literalism. This was stabilized by:
  * Implementing rigid **Few-Shot Prompting** (providing structural Q&A mapping examples).
  * Tuning decoding hyperparameters (`temperature=0.6`, `do_sample=True`).
  * Utilizing a `repetition_penalty=2.0` to mathematically forbid recursive loops.
* **UI Fallback Systems:** Implemented backend Python `while` loops to ensure the Matplotlib visual grid populates smoothly with default fallback items if the LLM generation process experiences early stopping.

---

## 💻 How to Run This Code

### Option 1: Run on Kaggle (Recommended)
1. Download the `Smart_Fashion_Engine.ipynb` file from this repository.
2. Upload the notebook to Kaggle.
3. Click **Add Data** and attach the `paramaggarwal/fashion-product-images-small` dataset.
4. Select **Run -> Restart & Run All**.
