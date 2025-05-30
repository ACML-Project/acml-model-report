# Conclusion 

In this project, we developed an LSTM-based binary text classifier to distinguish fake news from real news.

The articles were preprocessed via tokenization, stopword removal and lemmatization. Based on the dataset’s median length, the articles were truncated to a maximum of 256 tokens. This limitation improved performance by reducing noise and overfitting while preserving the most informative content in the news articles. 

Using a vocabulary derived from only the training set, the model was able to generalise efficiently, successfully understanding OOV words once tested on the validation and test datasets. The model architecture included a two-layer LSTM with 128 hidden units and a fully connected layer for binary classification. Regularization was applied using dropout, and weight decay.

Our model achieved near-perfect performance on both validation and test datassets, with an overall test accuracy of 99.93% and an F1 score of 1.00 for both classes. These results suggest the model successfully captured the linguistic and structural patterns distinguishing fake from real news.