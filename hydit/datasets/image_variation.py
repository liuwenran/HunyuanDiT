import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import PIL
from PIL import Image
from transformers import CLIPImageProcessor
import json
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import jsonlines
import copy
import cv2
from decord import VideoReader
from petrel_client.client import Client
from io import BytesIO

class ImageVariationDataset(Dataset):
    def __init__(self, 
                 data_config, 
                 img_size=(1024, 1024),
                 img_scale=(1.0, 1.0),
                #  img_ratio=(0.55, 0.57),
                 img_ratio=(0.9, 1.0),
                 drop_ratio=0.1,
                 backend=None,):
        super().__init__()
        self.backend = backend
        if backend == 'petreloss':
            self.client = Client(conf_path='/mnt/petrelfs/liuwenran/petreloss.conf')
            self.bucket_root = 'lol:s3://'

        self.img_size = img_size
        width, height = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.all_image_path = []
        for i in range(len(data_config.image_file)):
            data_lines = open(data_config.image_file[i], 'r').read().splitlines()
            image_path_lines = [os.path.join(data_config.image_root_path[i], data_line) for data_line in data_lines]
            self.all_image_path.extend(image_path_lines)

        print(f'Dataset size: {len(self.all_image_path)}')

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (height, width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                # transforms.RandomResizedCrop(
                #     (height, width),
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(
                    (height, width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                # transforms.RandomResizedCrop(
                #     (height, width),
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio


    def __getitem__(self, index):
        try:
            image_path = self.all_image_path[index]

            ### read ref image and target image and target pose
            if self.backend == 'petreloss':
                img_url = f'{self.bucket_root}{image_path}'
                img_bytes = self.client.get(img_url)
                assert(img_bytes is not None)
                img_mem_view = memoryview(img_bytes)
                img_array = np.frombuffer(img_mem_view, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)

            img = img.resize(self.img_size)

            state = torch.get_rng_state()
            tgt_img = self.augmentation(img, self.transform, state)

            clip_image = self.clip_image_processor(
                images=img, return_tensors="pt"
            ).pixel_values[0]


        except Exception:
            print(f'load error happens image_path {image_path} ')
            file = open('broken_vids.txt', 'a')
            line = image_path
            file.writelines([line + '\n'])
            file.flush()
            return self.__getitem__((index + 1) % len(self.all_image_path))

        image_meta_size = tuple(self.img_size) + tuple(self.img_size) + tuple((0, 0))
        image_meta_size = torch.tensor(np.array(image_meta_size)).clone().detach() 
        style = 0
        style = torch.tensor(np.array(style)).clone().detach()

        sample = dict(
            image_path=image_path,
            img=tgt_img,
            clip_images=clip_image,
            ind=index,
            image_meta_size=image_meta_size,
            style=style,
        )

        return sample

    def __len__(self):
        return len(self.all_image_path)

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/petrelfs/liuwenran/forks/HunyuanDiT/dataset/human_animation_yamls/stage1_image_variation.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    # client = Client(conf_path='/mnt/petrelfs/liuwenran/petreloss.conf')
    # bucket_root = 'lol:s3://'
    # image_path = 'private-dataset-pnorm/pexels-img/image/'
    # image_id = '00001f5a269f46f1191ac89306ba660b342db06c0f71ec6a778d98184dd999b3.jpg'
    # img_url = f'{bucket_root}{image_path}{image_id}'
    # img_bytes = client.get(img_url)
    # assert(img_bytes is not None)
    # img_mem_view = memoryview(img_bytes)
    # img_array = np.frombuffer(img_mem_view, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    dataset = ImageVariationDataset(data_config=cfg.data, img_size=(cfg.data.train_width, cfg.data.train_height), backend='petreloss')

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for step, batch in enumerate(train_dataloader):
        batch_index = batch['ind']
        pixel_values = batch["img"]
        img_path = batch['image_path']
        print(f'step {step}')
        import ipdb;ipdb.set_trace();
        # print(f'batch_index {batch_index} pixel_values shape {pixel_values.shape}')