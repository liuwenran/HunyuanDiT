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
from controlnet_aux import HEDdetector, OpenposeDetector
import cv2
from decord import VideoReader
from petrel_client.client import Client
from io import BytesIO

class HumanAnimationImageDataset(Dataset):
    def __init__(self, 
                 data_config, 
                 img_size=(512, 512),
                 control_type='pose',
                 img_scale=(1.0, 1.0),
                #  img_ratio=(0.55, 0.57),
                 img_ratio=(0.9, 1.0),
                 drop_ratio=0.1,
                 use_depth_enhance=False,
                 backend=None,
                 use_ref_pose_guider=False,
                 use_hand_depth=False):
        super().__init__()
        self.backend = backend
        if backend == 'petreloss':
            self.client = Client(conf_path='/mnt/petrelfs/liuwenran/petreloss.conf')
            self.bucket_root = 'liuwenran:s3://'

        self.sample_margin = data_config.sample_margin
        self.margin_strategy = data_config.margin_strategy
        self.load_depth = use_depth_enhance
        self.use_ref_pose_guider = use_ref_pose_guider
        self.use_hand_depth = use_hand_depth

        self.img_size = img_size
        width, height = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.control_type = control_type
        if self.control_type == 'hed':
            # self.hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
            from controlnet_aux import HEDdetector
            self.hed = HEDdetector.from_pretrained('/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96')
  
        self.all_video_path = []
        self.all_pose_path = []
        self.all_data_type = []
        if self.load_depth:
            self.all_depth_path = []
        if self.use_hand_depth:
            self.all_hand_depth_path = []
        for i in range(len(data_config.vid_file)):
            data_lines = open(data_config.vid_file[i], 'r').read().splitlines()
            video_path_lines = [os.path.join(data_config.video_root_path[i], data_line) for data_line in data_lines]
            pose_path_lines = [os.path.join(data_config.pose_root_path[i], data_line) for data_line in data_lines]
            self.all_video_path.extend(video_path_lines)
            self.all_pose_path.extend(pose_path_lines)
            if self.load_depth:
                depth_path_lines = [os.path.join(data_config.depth_root_path[i], data_line) for data_line in data_lines]
                self.all_depth_path.extend(depth_path_lines)
            if self.use_hand_depth:
                hand_depth_path_lines = [os.path.join(data_config.hand_depth_root_path[i], data_line) for data_line in data_lines]
                self.all_hand_depth_path.extend(hand_depth_path_lines)

            data_lines_type = [data_config.data_type[i]] * len(data_lines)
            self.all_data_type.extend(data_lines_type)

        print(f'Dataset size: {len(self.all_video_path)}')

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
            video_path = self.all_video_path[index]
            pose_path = self.all_pose_path[index]
            data_type = self.all_data_type[index]
            depth_path = None
            if self.load_depth:
                depth_path = self.all_depth_path[index]
            if self.use_hand_depth:
                hand_depth_path = self.all_hand_depth_path[index]

            tgt_image_ind, reference_image_ind = None, None
            if data_type == 'video':
                video_reader = VideoReader(video_path)
                kps_reader = VideoReader(pose_path)

                assert len(video_reader) == len(
                    kps_reader
                ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

                video_length = len(video_reader)
            elif data_type == 'shijuezhongguo_hengping_local':
                video_path = video_path + '.mp4'
                video_reader = VideoReader(video_path)

                frame_files = os.listdir(pose_path)
                frame_files = [os.path.join(pose_path, f) for f in frame_files]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_pose_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_pose_images.append(frame)
                assert len(video_reader) == len(
                    all_pose_images
                    ), f"{len(video_reader) = } != {len(all_pose_images) = } in {video_path}"
                video_length = len(video_reader)
            elif data_type == 'shijuezhongguo_hengping':
                # load origin video from ceph and load dwpose from ceph frames
                video_path = video_path + '.mp4'
                body = self.client.get(video_path)
                video_reader = VideoReader(BytesIO(body))
                
                file_iter = self.client.get_file_iterator(pose_path)
                frame_files = [self.bucket_root + k for k, v in file_iter]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_pose_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_pose_images.append(frame)
                
                assert len(video_reader) == len(
                        all_pose_images
                    ), f"{len(video_reader) = } != {len(all_pose_images) = } in {video_path}"
                
                video_length = len(video_reader)
            else:
                # load images path
                if self.backend == 'petreloss':
                    file_iter = self.client.get_file_iterator(video_path)
                    frame_files = [self.bucket_root + k for k, v in file_iter]
                else:
                    frame_files = os.listdir(video_path)
                    frame_files = [os.path.join(video_path, f) for f in frame_files]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_images.append(frame)
                
                # load pose path
                if self.backend == 'petreloss':
                    file_iter = self.client.get_file_iterator(pose_path)
                    frame_files = [self.bucket_root + k for k, v in file_iter]
                else:
                    frame_files = os.listdir(pose_path)
                    frame_files = [os.path.join(pose_path, f) for f in frame_files]
                frame_files.sort()
                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_pose_images = []
                for frame in frame_files:
                    frame_extension = os.path.splitext(frame)[1]
                    if frame_extension in IMAGE_EXTENSIONS:
                        all_pose_images.append(frame)

                # load depth path
                if self.load_depth:
                    frame_files = os.listdir(depth_path)
                    frame_files = [os.path.join(depth_path, f) for f in frame_files]
                    frame_files.sort()
                    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                    all_depth_images = []
                    for frame in frame_files:
                        frame_extension = os.path.splitext(frame)[1]
                        if frame_extension in IMAGE_EXTENSIONS:
                            all_depth_images.append(frame)
                
                if self.use_hand_depth:
                    frame_files = os.listdir(hand_depth_path)
                    frame_files = [os.path.join(hand_depth_path, f) for f in frame_files]
                    frame_files.sort()
                    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                    all_hand_depth_images = []
                    for frame in frame_files:
                        frame_extension = os.path.splitext(frame)[1]
                        if frame_extension in IMAGE_EXTENSIONS:
                            all_hand_depth_images.append(frame)

                assert len(all_images) == len(
                    all_pose_images
                ), f"{len(all_images) = } != {len(all_pose_images) = } in {video_path}"

                if self.load_depth:
                    assert len(all_images) == len(
                        all_depth_images
                    ), f"{len(all_images) = } != {len(all_depth_images) = } in {video_path}"

                video_length = len(all_images)
                if self.use_hand_depth:
                    video_length = len(all_hand_depth_images)

            ### get ref image and target image index
            tgt_image_ind = random.randint(0, video_length - 1)

            if self.margin_strategy == 'close':
                reference_image_ind = random.randint(-self.sample_margin, self.sample_margin) + tgt_image_ind
                if reference_image_ind < 0:
                    reference_image_ind = 0
                if reference_image_ind > video_length - 1:
                    reference_image_ind = video_length - 1
            elif self.margin_strategy == 'far':
                if tgt_image_ind + self.sample_margin < video_length:
                    reference_image_ind = random.randint(tgt_image_ind + self.sample_margin, video_length - 1)
                elif tgt_image_ind - self.sample_margin > 0:
                    reference_image_ind = random.randint(0, tgt_image_ind - self.sample_margin)
                else:
                    reference_image_ind = random.randint(0, video_length - 1)
            else:
                import sys
                sys.exit()

            ### read ref image and target image and target pose
            if self.use_hand_depth:
                tgt_image_name = all_hand_depth_images[tgt_image_ind].split('/')[-1].split('.')[0]
                reference_image_name = all_hand_depth_images[reference_image_ind].split('/')[-1].split('.')[0]
                tgt_image_path = os.path.join(video_path, tgt_image_name + '.png')
                reference_image_path = os.path.join(video_path, reference_image_name + '.png')
                tgt_image = Image.open(tgt_image_path).convert("RGB")
                reference_image = Image.open(reference_image_path).convert("RGB")
                hand_depth_image = Image.open(all_hand_depth_images[tgt_image_ind]).convert("RGB")
            else:
                if data_type == 'shijuezhongguo_hengping' or data_type == 'shijuezhongguo_hengping_local':
                    reference_image = video_reader[reference_image_ind]
                    reference_image = Image.fromarray(reference_image.asnumpy())
                    tgt_image = video_reader[tgt_image_ind]
                    tgt_image = Image.fromarray(tgt_image.asnumpy())
                elif self.backend == 'petreloss':
                    body = self.client.get(all_images[reference_image_ind])
                    reference_image = Image.open(BytesIO(body)).convert("RGB")
                    tgt_body = self.client.get(all_images[tgt_image_ind])
                    tgt_image = Image.open(BytesIO(tgt_body)).convert("RGB")
                else:
                    reference_image = Image.open(all_images[reference_image_ind]).convert("RGB")
                    tgt_image = Image.open(all_images[tgt_image_ind]).convert("RGB")
                if self.load_depth:
                    ref_depth_image = Image.open(all_depth_images[reference_image_ind]).convert("RGB")
                    tgt_depth_image = Image.open(all_depth_images[tgt_image_ind]).convert("RGB")

            control_image = None
            if self.control_type == 'canny':
                control_image = np.array(tgt_image)
                control_image = cv2.Canny(control_image, 100, 200)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            elif self.control_type == 'hed':
                control_image = self.hed(tgt_image)
            elif self.control_type == 'pose':
                if self.backend == 'petreloss':
                    body = self.client.get(all_pose_images[tgt_image_ind])
                    control_image = Image.open(BytesIO(body)).convert("RGB")
                else:
                    control_image = Image.open(all_pose_images[tgt_image_ind]).convert("RGB")
                if self.use_ref_pose_guider:
                    if self.backend == 'petreloss':
                        body = self.client.get(all_pose_images[reference_image_ind])
                        ref_pose_guider_image = Image.open(BytesIO(body)).convert("RGB")
                    else:
                        ref_pose_guider_image = Image.open(all_pose_images[reference_image_ind]).convert("RGB")
                if self.use_hand_depth:
                    control_image_path = os.path.join(pose_path, tgt_image_name + '.jpg')
                    control_image = Image.open(control_image_path).convert("RGB")


            control_image = control_image.resize(tgt_image.size)
            if self.use_ref_pose_guider:
                ref_pose_guider_image = ref_pose_guider_image.resize(reference_image.size)
            if self.use_hand_depth:
                hand_depth_image = hand_depth_image.resize(tgt_image.size)

            state = torch.get_rng_state()
            tgt_img = self.augmentation(tgt_image, self.transform, state)
            tgt_pose_img = self.augmentation(control_image, self.cond_transform, state)
            ref_img_vae = self.augmentation(reference_image, self.transform, state)
            if self.use_ref_pose_guider:
                ref_pose_guider_img = self.augmentation(ref_pose_guider_image, self.cond_transform, state)
            else:
                ref_pose_guider_img = ''
            if self.use_hand_depth:
                hand_depth_img = self.augmentation(hand_depth_image, self.transform, state)
            else:
                hand_depth_img = ''

            clip_image = self.clip_image_processor(
                images=reference_image, return_tensors="pt"
            ).pixel_values[0]
            if self.load_depth:
                ref_depth_img = self.augmentation(ref_depth_image, self.transform, state)
                tgt_depth_img = self.augmentation(tgt_depth_image, self.transform, state)
            else:
                ref_depth_img = ''
                tgt_depth_img = ''

        except Exception:
            print('load error happens')
            print(f'video_path {video_path} pose_path {pose_path} depth_path {depth_path} tgt_image_ind {tgt_image_ind} reference_image_ind {reference_image_ind}')
            file = open('broken_vids.txt', 'a')
            line = video_path.split('/')[-1].split('.')[0]
            file.writelines([line + '\n'])
            file.flush()
            return self.__getitem__((index + 1) % len(self.all_video_path))

        image_meta_size = tuple(self.img_size) + tuple(self.img_size) + tuple((0, 0))
        image_meta_size = torch.tensor(np.array(image_meta_size)).clone().detach() 
        style = 0
        style = torch.tensor(np.array(style)).clone().detach()

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
            ref_depth_img=ref_depth_img,
            tgt_depth_img=tgt_depth_img,
            ind=index,
            ref_pose=ref_pose_guider_img,
            hand_depth=hand_depth_img,
            image_meta_size=image_meta_size,
            style=style,
        )

        return sample

    def __len__(self):
        return len(self.all_video_path)

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/petrelfs/liuwenran/repos/HumanAnimation/configs/train/stage1_combine.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    dataset4 = HumanAnimationImageDataset(data_config=cfg.data4, img_size=(cfg.data4.train_width, cfg.data4.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, backend='petreloss')

    train_dataloader = torch.utils.data.DataLoader(
        dataset4, batch_size=cfg.data.train_bs, shuffle=False, num_workers=0
    )

    for step, batch in enumerate(train_dataloader):
        batch_index = batch['ind']
        pixel_values = batch["img"]
        print(f'step {step}')
        # print(f'batch_index {batch_index} pixel_values shape {pixel_values.shape}')