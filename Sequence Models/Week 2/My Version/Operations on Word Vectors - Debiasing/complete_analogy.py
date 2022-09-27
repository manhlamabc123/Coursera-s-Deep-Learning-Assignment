from cosine_similarity import *

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    # Convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embedding e_a, e_b and e_c
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.key() # Get list of word
    max_cosine_sim = -100 # Store max of cosine_sim
    best_word = None # Store the word coresponse to the max cosine_sim value

    # Loop over the whole word vector set
    for w in words:
        # Skip the word_c itself
        if w == word_c:
            continue

        # Compute cosine similarity between the vector (e_b - e_a) and the vector (w's vector representation - e_c)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        # This a simple find max algorithm
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word