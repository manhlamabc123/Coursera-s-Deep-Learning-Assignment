import torch
import numpy as np

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    """    
    seq = 1 - torch.from_numpy(np.equal(decoder_token_ids, 0))
    return seq[:, None, :]