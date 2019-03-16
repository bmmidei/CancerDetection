import csv
import math
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CancerDataset(Dataset):
    '''Dataset class to feed network with images and labels
    args:
        df (pandas df): dataframe containing two columns: image ids and
                        target labels
        image_dir (string): directory with the images
        transform (callable, optional): optional transform to be applied
                                        on a sample. should be 'none' for testing
    '''
    def __init__(self, df, image_dir, transform=None):
        self.labels = df.label      
        self.im_names = df.id
        self.image_dir = image_dir

        # Transform to apply to each image - note that all images will be
        # converted to tensors
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                            transforms.ToTensor()])
                            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extract image from file
        im_name = self.im_names[idx]
        im_path = os.path.join(self.image_dir, im_name + '.tif')
        im = Image.open(im_path)
        
        # Transform image
        im = self.transform(im)

        label = np.array(int(self.labels[idx]))

        sample = {'image': im.type(torch.cuda.FloatTensor),
                  'label': torch.from_numpy(label).type(torch.cuda.LongTensor)}

        return sample