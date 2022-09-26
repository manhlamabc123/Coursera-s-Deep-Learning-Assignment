from turtle import update
import numpy as np
from rnn_utils import *

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    # Retrieve parameters from "parameters"
    ## Forget-gate weight (Wf, bf)
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    ## Update-gate weight (Wu, bu)
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    ## Candiate weight (Wc, bc)
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    ## Output-gate weight (Wo, bo)
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    ## Prediction weight (Wy, by)
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt))

    # Compute values for forget_gate, update_gate, candiate_c, c_next, output_gate, a_next
    forget_gate = sigmoid(np.add(np.dot(Wf, concat), bf)) # forget_gate = sigmoid(Wf[a_prev, xt] + bf)
    update_gate = sigmoid(np.add(np.dot(Wu, concat), bu)) # update_gate = sigmoid(Wu[a_prev, xt] + bu)
    candiate_c = np.tanh(np.add(np.dot(Wc, concat), bc)) # candiate_c = tanh(Wc[a_prev, xt] + bc)
    c_next = forget_gate * c_prev + update_gate * candiate_c # element wise product
    output_gate = sigmoid(np.add(np.dot(Wo, concat), bo)) # output_gate = sigmoid(Wo[a_prev, xt] + bo)
    a_next = output_gate * np.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = softmax(np.add(np.dot(Wy, a_next), by))

    # Store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, forget_gate, update_gate, candiate_c, output_gate, xt, parameters)
    
    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    # Create list of cache
    caches = []

    # Retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # Initialize "a", "c" and "y"
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))

    # Loop over all time-steps
    for t in range(T_x):
        # Get 2D slide "xt" from 3D "X" at time step "t"
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:,:,t] = a_next
        # Save the value of the next cell state
        c[:,:,t] = c_next
        # Save the value of the prediction in y
        y[:,:,t] = yt
        # Add "cache" to "caches"
        caches.append(cache)

    # Store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches