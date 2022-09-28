import tensorflow as tf

def create_look_ahead_mask(sequence_length):
    '''
    Returns an lower triangular matrix filled with ones
    Arguments:
        sequence_length: matrix size
    Returns:
        mask: (size, size) tensor
    '''
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask