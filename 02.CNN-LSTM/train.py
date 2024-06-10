import gc
import os
import time
import pickle
import numpy as np

import torch
import torch.cuda
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms

from dataset import *
from dataloader import *
from models import *
from vocab import Vocab

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, __module_name: str, __global_name: str):
        if __module_name == '__main__':
            __module_name = __name__
        return super().find_class(__module_name, __global_name)

if __name__ == '__main__':
    try:
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("GPU device name: ", torch.cuda.get_device_name(0))

        # Setting cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        # Create model directory
        if not os.path.exists('./models_dir/'):
            os.makedirs('./models_dir/')
            
        # Image preprocessing, normalization for the pretrained resnet
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load vocabulary wrapper
        vocab_dir = storage_dir + 'vocabulary.pkl'
        with open(vocab_dir, 'rb') as f:
            unpickler = MyCustomUnpickler(f)
            vocabulary = unpickler.load()
            
        # Build data loader
        # resized_dir = storage_dir + 'resized'
        # anns_train_dir = storage_dir + 'annotations/captions_train2017.json'
        hdf5_file = storage_dir + 'hdf5/CustomCOCO.hdf5'
        custom_data_loader = get_loader(
            hdf5_file, vocabulary, transform, 256, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False
        )

        # Build the models
        encoder_model = CNNModel(64).to(device)
        decoder_model = LSTMModel(64, 128, len(vocabulary), 1).to(device)

        # Loss and optimizer
        loss_criterion = nn.CrossEntropyLoss()
        parameters = list(decoder_model.parameters()) + list(encoder_model.linear_layer.parameters()) + list(encoder_model.batch_norm.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        
        total_params = sum(p.numel() for p in parameters)
        print(f'Total number of parameters: {total_params}')
        
        # Gradient Scaler for mixed precision
        scaler = GradScaler()

        # Train the models
        start_time = time.time()
        total_num_steps = len(custom_data_loader)
        epochs = 5

        print('\n\nStart Training\n\n')
        for epoch in range(epochs):
            torch.cuda.empty_cache()
            gc.collect()
            for i, (imgs, caps, lens) in enumerate(custom_data_loader):                
                # Set mini-batch dataset
                imgs = imgs.to(device)
                caps = caps.to(device)
                tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]
                
                # Forward, backward and optimize
                optimizer.zero_grad()
                with autocast():
                    feats = encoder_model(imgs)
                    outputs = decoder_model(feats, caps, lens)
                    loss = loss_criterion(outputs, tgts)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                torch.cuda.empty_cache()
                gc.collect()
                
                # Print log info
                if i % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                        .format(epoch, epochs, i, total_num_steps, loss.item(), np.exp(loss.item())))
                    # elapsed_time_steps = 0

                # Save the model checkpoints
                if (i+1) % 100 == 0:
                    torch.save(decoder_model.state_dict(), os.path.join(
                        'models_dir/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                    torch.save(encoder_model.state_dict(), os.path.join(
                        'models_dir/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            
        end_time = time.time()
        elapsed = round((end_time - start_time) / 60, 2)
        print(f'Elapsed Time : {elapsed} mins.')
                    
    except KeyboardInterrupt:
        print('Training Interrupted.')
        
    finally:
        if 'custom_data_loader' in locals():
            del custom_data_loader
        torch.cuda.empty_cache()
        print('Resources cleaned up.')