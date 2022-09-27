from model import *
from utils import *

# Dataset and preprocessing
data = open('dinos.txt', 'r').read()
data = data.lower() # list of characters
chars = sorted(list(set(data))) # list of unique characters
data_size, vocab_size = len(data), len(chars) # number of characters and number of unique characters
char_to_ix = {ch:i for i, ch in enumerate(chars)} # dict - 'char': ix
ix_to_char = {i:ch for i, ch in enumerate(chars)} # dict - ix: 'char'

parameters, last_name = model(data.split('\n'), ix_to_char, char_to_ix, 22001, verbose=True)