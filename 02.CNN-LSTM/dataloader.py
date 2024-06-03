import os
import nltk
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence


class CustomCocoDataset(data.Dataset):
    """
    COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """
    def __init__(self, data_dir, coco_json_dir, vocabulary, transform=None):
        """
        Set the directory for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = data_dir
        self.coco_data = COCO(coco_json_dir)
        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform
        
    def __getitem__(self, idx):
        """
        Returns one data pair (image and caption).
        """
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]['caption']
        image_id = coco_data.anns[annotation_id]['image_id']
        image_dir = coco_data.loadImgs(image_id)[0]['file_name']
        
        image = Image.open(os.path.join(self.root, image_dir)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption(string) to word ids
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocabulary('<start>'))
        caption.extend([vocabulary(token) for token in word_tokens])
        caption.append(vocabulary('<end>'))
        ground_truth = torch.Tensor(caption)
        
        return image, ground_truth
    
    # SourceCode Git Commit!