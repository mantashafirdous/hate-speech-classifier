# Deep Learning-Based Text Classification for Hate Speech in Online Social Networks

## 📌 Overview

This project focuses on building a robust deep learning-based system to detect and classify hate speech in online social media platforms, specifically using Twitter data. The goal is to automatically identify hateful, offensive, and neutral speech to support content moderation efforts.

---

## 🧩 Problem Statement

The rise of social media has also led to the widespread proliferation of hate speech. This project aims to:

* **Binary Classification (Dataset 1)**: Distinguish between *Hate Speech* and *Non-Hate Speech*.
* **Multi-Class Classification (Dataset 2)**: Classify tweets into *Hate Speech*, *Offensive Speech*, and *Neutral Speech*.

---

## 🔍 Datasets Used

### 📁 Dataset 1: [Twitter Hate Speech Dataset](https://www.kaggle.com/datasets/vkrahul/twitter-hate-speech)

* **Samples**: 31,962 tweets
* **Classes**: Hate Speech (7.02%), Non-Hate Speech (92.98%)

### 📁 Dataset 2: [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

* **Samples**: 24,783 tweets
* **Classes**: Hate Speech, Offensive Language, Neither

---

## ⚙️ Workflow

1. **Data Preprocessing**: Tokenization, padding, and normalization
2. **Feature Extraction**:

   * GloVe Embeddings (100d)
   * TF-IDF (5000 features)
   * Transformer Embeddings (BERT, ALBERT, ELECTRA, etc.)
3. **Modeling**: CNN, LSTM, Bi-LSTM, MLP, SVM, XGBoost
4. **Evaluation**: Precision, Recall, F1-score (macro & weighted), ROC-AUC

---

## 🧠 Model Architectures

* **CNN (Single & Multi-gram filters)**
* **LSTM & cLSTM (Convolutional LSTM)**
* **Bi-LSTM**
* **MLP (Multi-Layer Perceptron)**
* **Transformer Models**:

  * BERT
  * ALBERT
  * ELECTRA
  * Small BERT

---

## 📊 Experimental Results

| Model           | Macro F1 Score | ROC-AUC  |
| --------------- | -------------- | -------- |
| TF-IDF + MLP    | **0.94**       | 0.97     |
| GloVe + Bi-LSTM | 0.89           | 0.93     |
| ELECTRA + MLP   | 0.87           | **0.98** |
| BERT + MLP      | **0.91**       | **0.98** |

> MLP with transformer embeddings (especially BERT and ELECTRA) showed the most consistent and accurate results.

---

## 🔬 Tools & Technologies

* **Languages**: Python
* **Libraries**: TensorFlow, Keras, Scikit-learn, XGBoost, Transformers (Hugging Face)
* **Embeddings**: GloVe, TF-IDF, BERT Variants
* **Datasets**: Kaggle

---

## 🚀 Future Work

* Integrate **multimodal detection** (images + text)
* Expand to **multilingual hate speech detection**
* Improve recall using advanced **data augmentation techniques**
* Scale up with **larger and diverse datasets**

---

## 👨‍🏫 Authors

* **Faizan Ahamad** – *21BCS047*
* **Mantasha Firdous** – *21BCS049*

**Supervisor**: Dr. Faiyaz Ahmad
**Institution**: Department of Computer Engineering, Jamia Millia Islamia (2025)

---

## 📜 References

Includes papers from IEEE Access, Springer, and major research works on hate speech detection using DL, ML, and transformer models. See the [presentation](https://github.com/mantashafirdous/hate-speech-classifier/blob/main/Presentation.pdf) for detailed references.
