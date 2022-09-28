import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
# from keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
from torch.nn import Embedding, MultiheadAttention, Dropout, LayerNorm
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification #???
from positional_encoding import *

pos_encoding = positional_encoding(50, 512)

print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('d')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()