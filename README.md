# Russian Literature Author Classification (Keras)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)

A Keras / TensorFlow project designed to classify Russian classical literature texts by author. The system uses neural networks with both Bag-of-Words and Embedding-based approaches to predict the author of a given text fragment.

---

## 📁 Project Structure

```bash
├── constants.py        # Project configuration constants
├── dataset.py          # Dataset download, preprocessing, tokenization, embedding matrix
├── model.py            # Model definition (Embedding + Dense layers)
├── train.py            # Training script for the model
├── graphs_example.py   # Optional: Visualization of training curves and confusion matrix
├── dataset/            # Auto-created folder for downloaded text data
└── README.md           # Project documentation
```

---
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Visualization](#visualization)
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Korny998/NLP-with-Natasha.git
cd NLP-with-Natasha
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

**To train the model and obtain predictions:**

```python
python train.py
```

The script will automatically:
1. Download the Russian literature dataset if it is not already present.
2. Preprocess the texts (tokenization, filtering, and windowing).
3. Build the embedding matrix using pre-trained Navec vectors.
4. Train the neural network on the processed sequences.
5. Save the model weights as pre_trained_model.h5.
6. Generate predictions on the test set.

## Model Architecture

1. Embedding Layer
- Converts word indices into 300-dimensional word vectors (from Navec pre-trained embeddings).
- Arguments:
    * input_dim=MAX_WORDS – Maximum number of words in the tokenizer vocabulary.
    * output_dim=EMBEDDING_DIM – Dimensionality of embedding vectors.
    * input_length=WIN_SIZE – Length of each input sequence (sliding window).
    * weights=[embedding_matrix] – Pre-trained word embeddings loaded from Navec.
- This layer is frozen (trainable=False) to keep embeddings fixed during training.

2. nBatchNormalization
- Normalizes the inputs to improve convergence.

3. Dense Layer
- Fully connected layer with 40 neurons and ReLU activation.
- Learns higher-level patterns from embedded sequences.

4. Dropout Layer
- Dropout rate = 0.6
- Helps prevent overfitting by randomly dropping neurons during training.

5. Second BatchNormalization
- Further normalizes data after dropout.

6. Flatten Layer
- Converts 2D output of previous layers into 1D for the final Dense layer.

7. Output Dense Layer
- len(CLASS_LIST) neurons with Softmax activation.
- Outputs probability distribution over all authors.

## Dataset

* Data Source

The dataset is automatically fetched from: https://storage.yandexcloud.net/academy.ai/russian_literature.zip

* Authors (Classes)

Dostoevsky, Tolstoy, Turgenev, Chekhov, Lermontov, Pushkin.

* Processing Pipeline

1. Tokenization: Texts are converted into sequences of integers using a tokenizer with MAX_WORDS=10,000.
2. Sliding Window: Each text is split into overlapping windows of length WIN_SIZE=1000 with step WIN_STEP=100.
3. Balancing: Ensures all classes have comparable sequence lengths by truncating longer sequences to MAX_SEQ=40,000.

# Visualization

The graphs_example.py module provides functions to evaluate the model:

* Training History: Accuracy and loss curves over epochs.
* Confusion Matrix: Shows percentage of correct vs incorrect predictions per author.