from utils import softmax
import numpy as np

def sample(parameters, char_to_ix, seed):
    # Retrieve parameters and relevent shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Create a zero vector x that can be used as one-hot vector
    x = np.zeros((vocab_size, 1)) # Representing the first character
    # Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices. This is the list which will contain the list of indices of the characters to generate
    indices = []

    # idx is the index of the element needed to be set to 1 in one-hot vector x
    idx = -1

    '''
    Loop over time-step t. At each time-step:
    - Sample a character from a probability distribution and append its index(idx) to "indices" list.
    - Loop 50 times (unlikely to reach this number with a well-trained model)
    '''
    counter = 0 # Increase each time loop
    newline_character = char_to_ix['\n'] # This should be 0

    while(idx != newline_character and counter != 50):
        # Forward propagate x
        a = np.tanh(np.add(np.dot(Waa, a_prev ), np.dot(Wax, x), b))
        z = np.add(np.dot(Wya, a), by)
        y = softmax(z)

        # Sample the index of a character within the vocabulary from the probability distribution y
        np.random.seed(counter + seed)
        idx = np.random.choice(range(len(y.ravel())), p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Overwrite the input x with the one that corresponds to the sampled index(idx)
        x = np.zeros((vocab_size, 1)) # Create a new x with 0 filled
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # Increase counter
        seed += 1
        counter += 1

    # In case we do reach 50 character, we need to append '\n' at the end
    if (counter == 50):
        indices.append(char_to_ix['\n'])

    return indices