import numpy as np
from rnn_utils import *

def rnn_cell_forward(xt, a_prev, parameters):
    # Retrive pararmeters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # Compute next activation state
    a_next = np.tanh(np.add(np.add(np.dot(Waa, a_prev), np.dot(Wax, xt)), ba)) # a^<t> = tanh(W_aa a^<t-1> + W_ax x^<t> + b_a)
    # Compute output of the current cell
    yt_pred = softmax(np.add(np.dot(Wya, a_next), by)) # hat(y)^<t> = softmax(W_ya a^<t> + b_y)

    # Store values needed for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    # Create list of cache
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # Initialize "a" and "y_pred" with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0

    # Loop over all time-step
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:,:,t] = a_next
        # Save the value of the prediction in y
        y_pred[:,:,t] = yt_pred
        # Add "cache" to "caches"
        caches.append(cache)
        
    # Store values need for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches