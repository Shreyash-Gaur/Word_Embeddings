
---

# Word Embeddings

This project involves working with word embeddings, which are dense vector representations of words that capture semantic relationships between them. By completing this project, you will gain an understanding of the various techniques used to create and utilize word embeddings in natural language processing (NLP) tasks.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [References](#references)

## Introduction

Word embeddings are a type of word representation that allows words to be represented as vectors in a continuous vector space. The key idea is that words that occur in similar contexts tend to be close to each other in the vector space. This project explores various techniques for generating and utilizing word embeddings, including the use of popular models like Word2Vec and GloVe.

## Objectives

The primary objectives of this project are to:

1. Preprocess a text corpus to create a clean dataset for training word embeddings.
2. Train word embedding models using different algorithms.
3. Evaluate the quality of the word embeddings using various techniques.
4. Visualize the embeddings to understand their properties and relationships.

## Dataset

The dataset used in this project is a text corpus that contains a large collection of sentences. This corpus is preprocessed to remove noise and prepare it for training word embedding models. Details about the dataset and preprocessing steps are included in the `Word_Embeddings.ipynb` notebook.

## Methodology

### Preprocessing

1. **Tokenization**: Splitting the text into individual words or tokens.
2. **Normalization**: Converting all words to lowercase and removing punctuation.
3. **Stop Words Removal**: Removing common words that do not contribute to the meaning (e.g., 'the', 'and').
4. **Lemmatization**: Reducing words to their base or root form.

### Model Training

- **Word2Vec**: Trains a neural network model to predict a word given its context (CBOW) or predict the context given a word (Skip-gram).
- **GloVe**: Generates word embeddings by factorizing the word co-occurrence matrix.

### Evaluation

- **Similarity Measures**: Evaluating the embeddings by measuring cosine similarity between word pairs.
- **Analogical Reasoning**: Testing the embeddings by solving analogy tasks (e.g., "man is to king as woman is to ?").

## Results

The trained word embeddings are evaluated using various metrics. The results section of the notebook provides detailed analysis and visualization, including:

- Cosine similarity scores for selected word pairs.
- t-SNE plots to visualize the embeddings in 2D space.
- Performance on analogy tasks.

![Image](/images/output.svg)

## Conclusion

This project demonstrates the process of generating and evaluating word embeddings. The trained embeddings capture semantic relationships between words, as evidenced by their performance on similarity and analogy tasks. These embeddings can be further utilized in various NLP applications like text classification, sentiment analysis, and more.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shreyash-Gaur/Word_Embeddings.git
   cd word_embeddings
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn gensim nltk matplotlib seaborn jupyter
   ```

3. **Run the Jupyter Notebook**:
   Open `Word_Embeddings.ipynb` in Jupyter Notebook or Jupyter Lab to explore the code, run the cells, and visualize the results.


## References

1. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781.
2. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 1532-1543).

---
