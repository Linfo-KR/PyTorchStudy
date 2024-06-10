import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from dataset import *
from train import MyCustomUnpickler
from models import CNNModel, LSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_file_dir, transform=None):
    img = Image.open(image_file_dir).convert('RGB')
    img = img.resize([224, 224], Image.Resampling.LANCZOS)
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
        
    return img

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load vocabulary wrapper
vocab_dir = storage_dir + 'vocabulary.pkl'
with open(vocab_dir, 'rb') as f:
    unpickler = MyCustomUnpickler(f)
    vocabulary = unpickler.load()
    
# Build models
encoder_model = CNNModel(128).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_model = LSTMModel(128, 256, len(vocabulary), 1)
encoder_model = encoder_model.to(device)
decoder_model = decoder_model.to(device)

# Load the trained model parameters
encoder_model.load_state_dict(torch.load('./models_dir/encoder-1-1000.ckpt'))
decoder_model.load_state_dict(torch.load('./models_dir/decoder-1-1000.ckpt'))

# Prepare an image
image_file_dir = './sample/sample_image.jpg'
img = load_image(image_file_dir, transform)
img_tensor = img.to(device)

# Generate an caption from the image
feat = encoder_model(img_tensor)
sampled_indices = decoder_model.sample(feat)
sampled_indices = sampled_indices[0].cpu().numpy()

# Convert word_ids to words
predicted_caption = []
for token_index in sampled_indices:
    word = vocabulary.i2w[token_index]
    predicted_caption.append(word)
    if word == '<end>':
        break
predicted_sentence = ' '.join(predicted_caption)

# Print out hte image and the generated caption

print(predicted_sentence)
img = Image.open(image_file_dir)
plt.imshow(np.asarray(img))