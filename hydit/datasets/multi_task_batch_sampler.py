import math
import torch
from torch.utils.data.sampler import RandomSampler
from omegaconf import OmegaConf
import argparse
from torch.utils.data.dataset import ConcatDataset
import random
import numpy as np

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, rank=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.rank = rank

    def __len__(self):
        data_len = 0
        for cur_dataset in self.dataset.datasets:
            data_len += math.floor(len(cur_dataset) / self.batch_size) * self.batch_size
        return  data_len

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        generator = torch.Generator()
        generator.manual_seed(self.rank)
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset, generator=generator)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        samples_to_grab = self.batch_size

        final_samples_list = []
        for i in range(self.number_of_datasets):
            cur_dataset_samples = []
            cur_batch_sampler = sampler_iterators[i]
            iter_times = math.floor(len(self.dataset.datasets[i]) / self.batch_size)
            for iter_ind in range(iter_times):
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                cur_dataset_samples.append(cur_samples)
            final_samples_list.extend(cur_dataset_samples)
        
        random.shuffle(final_samples_list)
        final_samples_list = np.array(final_samples_list)
        final_samples_list = final_samples_list.flatten()

        # print("final_samples_list:", final_samples_list)
        return iter(final_samples_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/petrelfs/liuwenran/repos/HumanAnimation/configs/train/stage1_combine.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    from human_animation_image import HumanAnimationImageDataset
    dataset1 = HumanAnimationImageDataset(data_config=cfg.data, img_size=(cfg.data.train_width, cfg.data.train_height), control_type=cfg.control_type)
    dataset2 = HumanAnimationImageDataset(data_config=cfg.data2, img_size=(cfg.data2.train_width, cfg.data2.train_height), control_type=cfg.control_type)

    concat_dataset = ConcatDataset([dataset1, dataset2])

    train_dataloader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=cfg.data.train_bs, sampler=BatchSchedulerSampler(concat_dataset, batch_size=cfg.data.train_bs), shuffle=False, num_workers=0
    )

    for step, batch in enumerate(train_dataloader):
        batch_index = batch['ind']
        pixel_values = batch["img"]
        print(f'batch_index {batch_index} pixel_values shape {pixel_values.shape}')

