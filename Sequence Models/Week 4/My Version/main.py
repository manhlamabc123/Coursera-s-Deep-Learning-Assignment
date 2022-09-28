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
from look_ahead_mask import *

pos_encoding = positional_encoding(50, 512)

