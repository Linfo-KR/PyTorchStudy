import h5py
import torch
import torch.utils.data as data
from nltk.tokenize import word_tokenize
from torchvision import transforms
from PIL import Image

class CustomCocoDataset(data.Dataset):
    def __init__(self, hdf5_file, vocabulary, transform=None):
        self.hdf5_file = hdf5_file
        self.vocabulary = vocabulary
        self.transform = transform
        with h5py.File(hdf5_file, 'r') as hdf5:
            self.num_samples = len(hdf5['images'])
        
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hdf5:
            image = hdf5['images'][idx]
            caption = hdf5['captions'][idx].decode('utf-8')
            
        image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption(string) to word ids
        tokens = word_tokenize(caption.lower())
        caption_ids = [self.vocabulary('<start>')]
        caption_ids.extend([self.vocabulary(token) for token in tokens])
        caption_ids.append(self.vocabulary('<end>'))
        caption_tensor = torch.Tensor(caption_ids)
        
        return image, caption_tensor
    
    def __len__(self):
        return self.num_samples

def collate_function(data_batch):
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)
    imgs = [transforms.ToPILImage()(img) for img in imgs]  # Convert images to PIL images
    imgs = torch.stack([transforms.ToTensor()(img) for img in imgs], 0)  # Convert PIL images to tensors
    cap_lens = [len(cap) for cap in caps]
    tgts = torch.zeros(len(caps), max(cap_lens)).long()
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i, :end] = cap[:end]
        
    return imgs, tgts, cap_lens

def get_loader(hdf5_file, vocabulary, transform, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
    dataset = CustomCocoDataset(hdf5_file, vocabulary, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate_function, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return data_loader