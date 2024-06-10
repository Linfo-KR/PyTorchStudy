import nltk
import pickle

from collections import Counter
from pycocotools.coco import COCO

from dataset import *

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
    """
    Build a Simple Vocabulary Wrapper.
    """
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the Captions.".format(i+1, len(ids)))
            
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')
    
    for i, token in enumerate(tokens):
        vocab.add_token(token)
    
    return vocab


if __name__ == '__main__':
    nltk.download('punkt')
    json_dir = storage_dir + 'annotations/captions_train2017.json'
    vocab = build_vocabulary(json=json_dir, threshold=4)
    vocab_dir = storage_dir + 'vocabulary.pkl'
    with open(vocab_dir, 'wb') as f:
        pickle.dump(vocab, f)
        
    print('Total Vocabulary Size: {}'.format(len(vocab)))
    print("Saved the Vocabulary Wrapper to '{}'".format(vocab_dir))