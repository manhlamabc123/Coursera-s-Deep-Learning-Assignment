import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
    '''
    Calculate the attention weights.
    Arguments:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Default to None.
    Returns:
        output: attention_weights
    '''
    matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += ((1 - mask) * -1e9)
    pass