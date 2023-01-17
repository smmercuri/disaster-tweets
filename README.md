# Disaster Tweets

This repository contains preprocessing and modelling for the 'Natural Language Processing with Disaster Tweets' Kaggle challenge https://www.kaggle.com/competitions/nlp-getting-started/overview.

## Models

1. RNN

The RNN-based model includes text embedding and bidirectional LSTM layers on the `'text'` column before being fed into stacked densely-connected layers and a sigmoid activation layer.
Additional text-based metadata layers, categorical and numerical features can also be fed in as parallel neural networks, being concatenated to the text RNN
model before the densely connected layers.

Tested with an accuracy of 77% on the test set, making use of the `'text'` column and using the `'keyword'` column as a categorical features.

## Running Experiments

1. Download the data from LINK and place in `disaster-tweets/data/`.
2. Perform global preprocessing by running `python -m disaster_tweets.preprocessing.main`. This will perform standard text preprocessing.
3. Run `python -m disaster_tweets.main` to train the RNN model. 

Model features and parameters can be configured in `disaster_tweets/config.py`. RNN architecture can be configured within the `build_RNN` function in 
`disaster_tweets/main.py`.
