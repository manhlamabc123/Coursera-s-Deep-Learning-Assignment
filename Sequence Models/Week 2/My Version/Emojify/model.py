import numpy as np
from sentence_to_avg import *
from emo_utils import *

def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    any_word = list(word_to_vec_map.keys())[0]

    # Defind number of training examples
    m = Y.shape[0] # Number of training examples
    n_y = len(np.unique(Y)) # Number of classes
    n_h = word_to_vec_map[any_word].shape[0] # Dimensions of the GloVe vectors

    # Initilalize parameters using Xavier initializaton
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y, ))

    # Conver Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C = n_y)

    # Opitmization loop
    for t in range(num_iterations):
        cost = 0
        dW = 0
        db = 0

        for i in range(m):
            # Average the word vectors of the words from the i-th training examples
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward pass
            z = np.add(np.dot(W, avg), b)
            a = softmax(z)

            # Add the cost
            cost += Y_oh * a

            # Backward
            dz = a - Y_oh[i]
            dW += np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db += dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py
    
    return pred, W, b