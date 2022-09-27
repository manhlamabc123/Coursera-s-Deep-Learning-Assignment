from cmath import isclose
import numpy as np

def cosine_similarity(u, v):
    # if u = [0, 0] and v = [0, 0]
    if np.all(u == v):
        return 1

    # Compute the dot product between u and v
    dot = np.dot(u, v)
    # Compute the L2 norm of u, v ( sqrt(x1^2 + x2^2 + ... + xn^2) )
    norm_u = np.sqrt(np.sum(u**2)) 
    norm_v = np.sqrt(np.sum(v**2))

    # Avoid division by 0, check if product of 2 norm close to 0
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0

    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity