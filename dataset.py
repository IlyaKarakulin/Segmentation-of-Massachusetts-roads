import os
from PIL import Image
from pathlib import Path
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask

class SegmentDataset(data.Dataset):
    def __init__(self, path, mode='train'):
        self.size = 1200
        # self.crop_size = 512
        self.path_img = Path(path)
        self.path_mask = Path(path + '_labels')
        self.mode = mode
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        
        images_dict = {p.stem: p for p in self.path_img.rglob('*') if p.suffix.lower() in IMAGE_EXTS}
        masks_dict = {p.stem: p for p in self.path_mask.rglob('*') if p.suffix.lower() in IMAGE_EXTS}

        all_keys = sorted(set(images_dict) & set(masks_dict))
        
        self.images = [images_dict[k] for k in all_keys]
        self.masks  = [masks_dict[k] for k in all_keys]
        self.length = len(self.images)

        self.resize_img = tfs_v2.Resize(
            (self.size, self.size), 
            interpolation=tfs_v2.InterpolationMode.LANCZOS
        )
        self.resize_mask = tfs_v2.Resize(
            (self.size, self.size), 
            interpolation=tfs_v2.InterpolationMode.NEAREST
        )
        
        self.to_tensor = tfs_v2.Compose([
            tfs_v2.ToImage(),
            tfs_v2.ToDtype(torch.float32, scale=True)
        ])
        
        if mode == 'train':
            self.joint_transforms = tfs_v2.Compose([
                tfs_v2.RandomHorizontalFlip(p=0.5),
                tfs_v2.RandomVerticalFlip(p=0.5),
                tfs_v2.RandomAffine(
                    degrees=30,
                    shear=10,
                    fill=0
                ),
                # tfs_v2.RandomCrop(
                #     size=(self.crop_size, self.crop_size),
                #     pad_if_needed=True
                # )
            ])
            
            self.color_transforms = tfs_v2.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.1
            )

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')

        img = self.resize_img(img)
        mask = self.resize_mask(mask)
        
        img = TVImage(img)
        mask = TVMask(mask)
        
        if self.mode == 'train':
            img, mask = self.joint_transforms(img, mask)
            img = self.color_transforms(img)
        
        img = self.to_tensor(img)
        mask = mask.to(torch.float32)
        mask = torch.where(mask >= 250, 1.0, 0.0)

        return img, mask

    def __len__(self):
        return self.length