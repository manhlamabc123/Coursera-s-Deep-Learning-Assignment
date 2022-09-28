import numpy as np

def sentence_to_avg(sentence, word_to_vec_map):
    # Get a valid word contained in the word_to_vec_map
    any_word = list(word_to_vec_map.keys())[0]

    # Split sentence
    words = sentence.lower().split()

    # Initialize the average word vector
    avg = np.zeros(word_to_vec_map[any_word].shape)

    # Initialize count
    count = 0

    # Compute average the word vectors
    for w in words:
        if w in word_to_vec_map.keys():
            avg += word_to_vec_map[w]
            count += 1
    if count > 0:
        avg = avg/count

    return avg