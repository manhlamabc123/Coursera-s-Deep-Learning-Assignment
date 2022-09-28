import numpy as np
import tensorflow as tf

def get_angles(pos, k , d):

    i = k // 2
    angles = pos / (10000 ** (2 * i / d))

    return angles

def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # Initialize a matrix angle_ras of all the angles
    angles_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

    # Apply sin to even indices in the array; 2i
    angles_rads[:, 0::2] = np.sin(angles_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angles_rads[:, 1::2] = np.cos(angles_rads[:, 1::2])

    pos_encoding = angles_rads[np.newaxis, ...]
    
    # return tf.cast(pos_encoding, dtype=tf.float32)
    return tf.cast(pos_encoding, dtype=tf.float32)