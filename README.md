# Fake News Detection with UPFD Dataset

This repository contains code for fake news detection using the UPFD (User Preference-aware Fake News Detection) dataset. The UPFD dataset is a benchmark dataset for fake news detection that includes news propagation graphs extracted from Twitter. You can find more information about the dataset at [https://paperswithcode.com/dataset/upfd](https://paperswithcode.com/dataset/upfd).

## Dataset

The UPFD dataset consists of news propagation graphs extracted from Twitter. The source and raw data can be found at [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet). The preprocessing steps are described in the paper: [https://arxiv.org/pdf/2104.12259.pdf](https://arxiv.org/pdf/2104.12259.pdf).

The dataset is split into train and test sets. The node features are represented by a combination of Spacy Word2Vec embeddings and profile features.

## Models

Two models are implemented in this repository:

1. **GNN (Graph Neural Network)**: A graph neural network model that uses graph convolutions and pooling to learn node embeddings and make predictions.

2. **GNNWithAttentionFusion**: An extension of the GNN model that incorporates attention-based fusion of node embeddings and news embeddings.

## Training and Evaluation

The code provides a `Trainer` class that handles the training and evaluation of the models. The trainer is initialized with the model, loss function, optimizer, and device (CPU or GPU).

The training process is performed using the `train` method of the trainer, which iterates over the training data loader and updates the model parameters.

Evaluation is done using the `test` method of the trainer, which computes the test loss, accuracy, and F1 score on the test data loader.

## Results

The repository includes code to visualize the training and evaluation results. The loss, accuracy, and F1 score are plotted over epochs for different models and learning rates.

Additionally, t-SNE (t-Distributed Stochastic Neighbor Embedding) is used to visualize the learned node embeddings in a 2D space.

## Usage

To run the code, make sure you have the required dependencies installed. You can install them using the following command:

```
pip install -r requirements.txt
```

The main code is provided in the `main.ipynb` notebook. You can run the notebook cells sequentially to load the dataset, train the models, and visualize the results.

## Acknowledgments

The code in this repository is mainly based on the UPFD colab tutorial available at [https://colab.research.google.com/drive/1ZVZdehPPod6o4sF64QZa8I3NoSOH8MmC?usp=sharing](https://colab.research.google.com/drive/1ZVZdehPPod6o4sF64QZa8I3NoSOH8MmC?usp=sharing).

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, provide feedback, or report any issues you encounter while using this code.
