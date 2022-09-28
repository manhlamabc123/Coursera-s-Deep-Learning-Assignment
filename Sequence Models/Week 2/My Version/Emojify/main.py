from model import *
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

# Read data
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# Get the length of the longest string
max_length = len(max(X_train, key=len).split())

# Turn y from (m, 1) to (m, 5)
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# Load pretrained Word Embedding
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# Train on training set
np.random.seed(1)
pred, W, b = model(X_train, Y_train, word_to_vec_map)

# Test on Test set
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)
print(f"Training set accuracy: {pred_train}")
print(f"Training set accuracy: {pred_test}")

# Test
X_my_sentences = np.array(["i cherish you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)