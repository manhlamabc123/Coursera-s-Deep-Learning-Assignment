from utils import *
from gradient_clipping import *

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    # Forward
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Backpropagate
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Gradients clipping
    gradients = clip(gradients, max_value=5)

    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X) - 1]