# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pvml
import seaborn as sns

# Functions from SC_functions.py
def show_spectrogram(features, labels, classes, word, cbar=True):
    '''Returns the spectrogram of the first occurrence of word in features matrix'''
    row_index = np.where(classes == word)[0][0]
    row_index = np.where(labels == row_index)[0][0]
    print('row index = ', row_index)
    spectro = features[row_index,:].reshape(20, 80)
    plt.imshow(spectro, cmap = 'hot', aspect='auto')
    if cbar:
        plt.colorbar()
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title('Spectrogram of ' + word)

def show_spectrogram_multiple(features, labels, classes, words):
    '''Returns the spectrogram of the first occurrence of each word of words in the features matrix'''
    fig, axs = plt.subplots(1, len(words), figsize=(20, 5))
    sm = plt.cm.ScalarMappable(cmap='hot')
    sm.set_array(features)
    for w, ax in zip(words, axs):
        row_index = np.where(classes == w)[0][0]
        row_index = np.where(labels == row_index)[0][0]
        spectro = features[row_index,:].reshape(20, 80)
        ax.imshow(spectro, cmap = 'hot', aspect='auto')
        ax.set_title('word: %s' % w)
    fig.colorbar(sm, ax=axs)
    plt.show()

# Other normalization and utility functions
def mean_var_normalize(train_features, test_features):
    u = train_features.mean(0)
    sigma = train_features.std(0)
    train_features = (train_features - u) / sigma
    test_features = (test_features - u) / sigma
    return train_features, test_features

def min_max_normalize(train_features, test_features):
    min = train_features.min(0)
    max = train_features.max(0)
    train_features = (train_features - min) / (max - min)
    test_features = (test_features - min) / (max - min)
    return train_features, test_features

def max_abs_normalize(train_features, test_features):
    max = np.abs(train_features).max(0)
    train_features = train_features / max
    test_features = test_features / max
    return train_features, test_features

def whitening_normalize(train_features, test_features):
    mu = train_features.mean(0)
    sigma = np.cov(train_features.T)
    evals, evecs = np.linalg.eigh(sigma)
    w = evecs / np.sqrt(evals)
    train_features = (train_features - mu) @ w
    test_features = (test_features - mu) @ w
    return train_features, test_features

def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100

def confusion_matrix(Y, predictions, labels, show=False, rnorm=True):
    classes = Y.max() + 1
    cm = np.zeros((classes, classes), dtype=int)
    for i in range(classes):
        mask = (Y == i)
        counts = np.bincount(predictions[mask], minlength=classes)
        if rnorm:
            cm[i, :] = 100 * counts / max(1, counts.sum())
        else:
            cm[i, :] = counts
    if show:
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    return cm

# Main code from the notebook

# DATA LOADING
classes = np.loadtxt('speech-comands/classes.txt', dtype=str)
test_names = np.loadtxt('speech-comands/test-names.txt', dtype=str)
test_data = np.load('speech-comands/test.npz')
test_Y = test_data['arr_1']
test_X = test_data['arr_0']

train_names = np.loadtxt('speech-comands/train-names.txt', dtype=str)
train_data = np.load('speech-comands/train.npz')
train_Y = train_data['arr_1']
train_X = train_data['arr_0']

validation_names = np.loadtxt('speech-comands/validation-names.txt', dtype=str)
validation_data = np.load('speech-comands//validation.npz')
validation_Y = validation_data['arr_1']
validation_X = validation_data['arr_0']

# SHAPE EXPLORATION
print('test_X shape: ', test_X.shape)
print('train_X shape: ', train_X.shape)
print('validation_X shape: ', validation_X.shape)

# DATA EXPLORATION
show_spectrogram_multiple(train_X, train_Y, classes, ['backward', 'house', 'go'])

words = [w for w in classes]
counters = np.bincount(train_Y)
plt.figure(figsize=(15, 5))
plt.bar(words, counters)
plt.xticks(rotation=90)
plt.xlabel('Words', fontsize=12)
plt.ylabel('# Occurrences', fontsize=12)
plt.title('Occurrences for each word')
plt.show()
# TRAINING
net = pvml.MLP([1600, 35])
train_X_norm, test_X_norm = mean_var_normalize(train_X, test_X)
train_acc, test_acc = [], []

for epoch in range(100):
    # Here we use 'lr' instead of 'learning_rate'

    print("Starting training at epoch 0...")
    net.train(train_X_norm, train_Y, lr=1e-4)
    print("Completed training at epoch 0...")

    if epoch % 10 == 0:
        train_acc.append(accuracy(net, train_X_norm, train_Y))
        test_acc.append(accuracy(net, test_X_norm, test_Y))
        print(f'Epoch {epoch}: Train Acc: {train_acc[-1]}, Test Acc: {test_acc[-1]}')


# CONFUSION MATRIX
predictions = net.inference(test_X_norm)[0]
cm = confusion_matrix(test_Y, predictions, classes, show=True)

