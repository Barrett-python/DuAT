import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.image_root = image_root
        self.gt_root = gt_root
        self.samples   = [name for name in os.listdir(image_root) if name[0]!="."]
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(352, 352, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
#            A.RandomRotate90(p=0.2),
            ToTensorV2()
        ])
        
        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.image_root+'/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        name2  = self.color1[idx%len(self.color1)] if np.random.rand()<0.7 else self.color2[idx%len(self.color2)]
        image2 = cv2.imread(self.image_root+'/'+name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean , std  = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0,1), keepdims=True), image2.std(axis=(0,1), keepdims=True)
        image = np.uint8((image-mean)/std*std2+mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask  = cv2.imread(self.gt_root+'/'+name, cv2.IMREAD_GRAYSCALE)/255.0
        pair  = self.transform(image=image, mask=mask)
        
        return pair['image'], pair['mask']

    def __len__(self):
        return len(self.samples)


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
                                 ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
