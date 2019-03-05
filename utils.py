import csv
import math
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, image_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(csv_file, 'r') as fo:
            reader = csv.reader(fo)
            next(reader, None)
            self.labels = {row[0]:row[1] for row in reader}
        self.labels
        self.images = [x for x in self.labels.keys()]
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        im_name = self.images[idx]
        im_path = os.path.join(self.image_dir, im_name + '.tif')
        im = Image.open(im_path)
        imarray = np.array(im)

        label = np.array(int(self.labels[im_name]))
        sample = {'image': torch.from_numpy(imarray).type(torch.FloatTensor),
                  'label': torch.from_numpy(label).type(torch.LongTensor)}

        #if self.transform:
        #    sample = self.transform(sample)

        return sample