import copy
import numpy as np

def clip(gradients, max_value):
    gradients = copy.deepcopy(gradients) # ???

    dWaa = gradients['dWaa']
    dWax = gradients['dWax']
    dWya = gradients['dWya']
    db = gradients['db']
    dby = gradients['dby']

    # clip to mitigate exploding gradients
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -max_value, max_value, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients