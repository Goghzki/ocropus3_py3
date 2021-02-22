import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import numpy as np
import scipy.ndimage as ndi


class DocLayoutDataset(Dataset):
    """Doc Layout dataset."""

    def __init__(self, img_dir, label_dir, dilate_mask = (30,150), mask_background = 0.0, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.dilate_mask = dilate_mask
        self.mask_background = mask_background
        self.transform = transform
        self.train_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.train_list[idx]
        img_name = os.path.join(self.img_dir,name)
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)/255.
        label_name = os.path.join(self.label_dir,name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)/255.
        mask = ndi.maximum_filter(image, self.dilate_mask)
        mask = np.maximum(mask, self.mask_background)
        H, W = image.shape
        assert H, W == label.shape
        image = image.reshape(H, W, 1)
        label = label.reshape(H, W, 1)
        mask = mask.reshape(H, W, 1)
        sample = {'name':name, 'image': image, 'label': label, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name, image, label, mask = sample['name'] ,sample['image'], sample['label'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {'name': name,
                'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label),
                'mask': torch.FloatTensor(mask)}

if __name__ == "__main__":
    data = DocLayoutDataset("./image","./label", transform = transforms.Compose([ToTensor()]))
    for i in range(3):
        sample = data[i]

        print(i, sample['name'], sample['image'].shape, sample['label'].shape, sample['mask'].shape)
    dataloader = DataLoader(data, batch_size=2, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['name'])