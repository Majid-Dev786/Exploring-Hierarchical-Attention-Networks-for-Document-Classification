
# Exploring Hierarchical Attention Networks for Document Classification

## Description
This project implements a Hierarchical Attention Network (HAN) model for the purpose of document classification. 

Utilizing advanced NLP techniques and TensorFlow, it demonstrates how attention mechanisms in deep learning can enhance the interpretability and performance of models in understanding and categorizing text data.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Real-World Application Scenarios](#real-world-application-scenarios)

## Installation
To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Sorena-Dev/Exploring-Hierarchical-Attention-Networks-for-Document-Classification.git
cd Exploring-Hierarchical-Attention-Networks-for-Document-Classification
pip install numpy tensorflow
```

## Usage
To use the Hierarchical Attention Network model for document classification, follow these steps:

1. Prepare your dataset of documents and their corresponding labels.
2. Use the provided script to tokenize and pad your text data.
3. Create an instance of the HAN model by calling `hierarchical_attention_model`.
4. Train the model on your dataset.
5. Use the trained model to predict labels for new, unseen documents.

Example usage is demonstrated in the provided Python script, illustrating the model's training and prediction phases.

## Features
- **Hierarchical Attention Mechanism**: Implements a two-level attention model capable of attending to both words and sentences for document classification.
- **Bidirectional LSTM Layers**: Utilizes bidirectional LSTMs to capture context from both directions of the text.
- **Customizable Architecture**: Allows for easy adjustments to the number of sentences, words, and embedding dimensions to suit different datasets.
- **Real-World Data Application**: Includes examples on how to apply the model to real-world datasets for meaningful insights.

## Real-World Application Scenarios
- **Sentiment Analysis**: Classifying documents, such as movie or product reviews, into positive or negative sentiments.
- **Topic Identification**: Automatically identifying the topic or category of articles and papers.
- **Customer Feedback Analysis**: Understanding customer feedback by categorizing comments or reviews into various aspects or sentiments.

