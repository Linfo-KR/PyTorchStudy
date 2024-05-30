import os
import nltk
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO


nltk.download('punkt')


class Vocab(object):
    """
    Simple Vocabulary Wrapper.
    """
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0
        self.add_token('<unk>')
        
    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
    
    def __len__(self):
        return(len(self.w2i))
    
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1
            
def build_vocabulary(json, threshold):
    """_summary_

    Args:
        json (_type_): _description_
        threshold (_type_): _description_
    """