import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import torchvision.transforms.functional as F
import pycocotools.mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DistanceDataset(Dataset):
    def __init__(self, data_dir, json_path, transform=None, rgb=True, depth=True, resize=(360, 640), cls_bin_center=None, distance_scale=1., min_distance=-1., max_distance=100000):
        # distance_scale is used to scale to different units (e.g. m, cm, mm)
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, 'images')
        self.depth_dir = os.path.join(data_dir, 'depths')
        self.use_rgb = rgb
        self.use_depth = depth
        self.resize = resize
        self.transform = transform
        if cls_bin_center is not None:
            self.cls_bin_center = torch.tensor(cls_bin_center, dtype=torch.float32)
        else:
            self.cls_bin_center = None
        self.distance_scale = distance_scale
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        print(f"Number of samples: {len(self.samples)}")

        print(f"Apply data filtering: Min distance: {min_distance}, Max distance: {max_distance}")
        self.samples = [s for s in self.samples if s['normalized_answer'] >= min_distance and s['normalized_answer'] <= max_distance]
        print(f"Number of samples after filtering: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def decode_mask(self, rle):
        rle['counts'] = rle['counts'].encode('utf-8')
        return mask_utils.decode(rle).astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        components = []

        if self.use_rgb:
            img_path = os.path.join(self.rgb_dir, item['image'])
            try:
                rgb = Image.open(img_path).convert('RGB')
                rgb = F.resize(rgb, self.resize)
                rgb = np.array(rgb).astype(np.float32) / 255.0
                components.append(rgb)
            except Exception as e:
                print(f"Warning: Failed to load RGB image {img_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))

        if self.use_depth:
            depth_path = os.path.join(self.depth_dir, item['image'].replace('.png', '_depth.png'))
            try:
                depth = Image.open(depth_path)
                depth = F.resize(depth, self.resize, interpolation=Image.BILINEAR)
                depth = np.array(depth).astype(np.float32) / 255.0
                components.append(depth[..., None])
            except Exception as e:
                print(f"Warning: Failed to load depth image {depth_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))

        try:
            mask_a = self.decode_mask(item['rle'][0])
            mask_b = self.decode_mask(item['rle'][1])
            mask_a = Image.fromarray(mask_a)
            mask_b = Image.fromarray(mask_b)
            mask_a = np.array(F.resize(mask_a, self.resize, interpolation=Image.NEAREST)).astype(np.float32)
            mask_b = np.array(F.resize(mask_b, self.resize, interpolation=Image.NEAREST)).astype(np.float32)
            components.append(mask_a[..., None])
            components.append(mask_b[..., None])
        except Exception as e:
            print(f"Warning: Failed to decode or resize masks: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Stack input and convert to tensor
        input_tensor = np.concatenate(components, axis=-1)  # H x W x C
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)  # C x H x W
        distance_cls_idx = None

        if self.cls_bin_center is not None:
            distance_cls_idx = torch.argmin(torch.abs(self.cls_bin_center - item['normalized_answer']))
            distance = item['normalized_answer'] - self.cls_bin_center[distance_cls_idx] 
        else:
            distance = torch.tensor(item['normalized_answer']*self.distance_scale, dtype=torch.float32) 

        if self.transform:
            input_tensor = self.transform(input_tensor)

        if distance_cls_idx is not None:
            return input_tensor, distance, distance_cls_idx
        else:
            return input_tensor, distance