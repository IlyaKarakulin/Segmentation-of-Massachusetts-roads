from PIL import Image
from pathlib import Path
import glob

import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask


class MassachusettsRoadsDataset(data.Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode

        self.path_img = path + "/*.tiff"
        self.path_mask = path + '_labels' + "/*.tif"
        
        # IMAGE_EXTS = {'.tiff', '.tif', '.png'}
        # self.images = {p for p in self.path_img.rglob('*') if p.suffix.lower() in IMAGE_EXTS}
        # self.masks = {p for p in self.path_mask.rglob('*') if p.suffix.lower() in IMAGE_EXTS}

        self.images = glob.glob(self.path_img)
        self.masks = glob.glob(self.path_mask)
        self.images.sort()
        self.masks.sort()

        self.length = len(self.images)
        

        if mode == 'train':
            self.joint_transforms = tfs_v2.Compose([
                tfs_v2.Resize((self.size, self.size)),
                tfs_v2.RandomChoice([
                    tfs_v2.RandomRotation(degrees=[0, 0]),
                    tfs_v2.RandomRotation(degrees=[90, 90]),
                    tfs_v2.RandomRotation(degrees=[180, 180]),
                    tfs_v2.RandomRotation(degrees=[270, 270]),
                ]),

                tfs_v2.RandomAffine(
                    degrees=(-90, 90),
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    fill=0
                ),
                tfs_v2.RandomHorizontalFlip(p=0.5),
                tfs_v2.RandomVerticalFlip(p=0.5),
            ])
            
            self.color_transforms = tfs_v2.Compose([
                tfs_v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])

        self.resize = tfs_v2.Resize((1488, 1488))
        self.size = 1488
        
        self.to_tensor = tfs_v2.Compose([
            tfs_v2.ToImage(),
            tfs_v2.ToDtype(torch.float32, scale=True)
        ])
        
        self.normalize = tfs_v2.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )


    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        mask = Image.open(self.masks[item]).convert('L')

        img = TVImage(img)
        mask = TVMask(mask)

        if self.mode == 'train':
            img, mask = self.joint_transforms(img, mask)
            img = self.color_transforms(img)
        elif self.mode == 'test':
            img, mask = self.resize(img, mask)

        
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        mask = mask.to(torch.float32) / 255.0
        mask = torch.where(mask >= 0.5, 1.0, 0.0)
        

        return img, mask

    
    def __len__(self):
        return self.length
