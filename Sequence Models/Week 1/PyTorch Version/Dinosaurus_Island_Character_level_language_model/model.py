from optimize import optimize
from sample import *
from utils import *
import numpy as np

def model(data_x, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27, verbose=False):
    # Retrieve n_x, n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (This is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples)
    examples = [x.strip() for x in data_x]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization Loop
    for j in range(num_iterations):
        # Make sure index in range 0 to len(examples)
        idx = j % len(examples)

        # set the input X
        single_example = idx
        single_example_chars = [c for c in examples[single_example]]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None] + single_example_ix

        # Set the labels Y
        ix_newline = [char_to_ix['\n']]
        Y = X[1:] + ix_newline

        # Perform one optimization step
        current_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # debug statements to aid in correctly forming X, Y
        if verbose and j in [0, len(examples) -1, len(examples)]:
            print("j = " , j, "idx = ", idx,) 
        if verbose and j in [0]:
            print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")

        # Keep the loss smooth
        loss = smooth(loss, current_loss)

        # Every 2000 iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            print(f"Iteration: {j}, Loss: {loss}\n")

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_dino_name = get_sample(sampled_indices, ix_to_char)
                print(last_dino_name.replace('\n', ''))

                seed += 1
            
            print('\n')
    
    return parameters, last_dino_name