import gc
import json
import os
import random
import sys
import time
from functools import partial
from glob import glob
from pathlib import Path
import numpy as np

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer
from torchvision.transforms import functional as TF
from diffusers.models import AutoencoderKL
from transformers import BertModel, BertTokenizer, logging as tf_logging

from hydit.config import get_args
from hydit.constants import VAE_EMA_PATH, TEXT_ENCODER, TOKENIZER, T5_ENCODER
from hydit.lr_scheduler import WarmupLR
from hydit.data_loader.arrow_load_stream import TextImageArrowStream
from hydit.diffusion import create_diffusion
from hydit.ds_config import deepspeed_config_from_args
from hydit.modules.ema import EMA
from hydit.modules.fp16_layers import Float16Module
from hydit.modules.models import HUNYUAN_DIT_MODELS, HunYuanDiT
from hydit.modules.posemb_layers import init_image_posemb
from hydit.utils.tools import create_logger, set_seeds, create_exp_folder, model_resume, get_trainable_params
from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from peft import LoraConfig, get_peft_model
from hydit.modules.pose_guider import PoseGuider
from hydit.modules.mutual_self_attention import ReferenceAttentionControl
from hydit.datasets.human_animation_image import HumanAnimationImageDataset
from hydit.datasets.multi_task_batch_sampler import BatchSchedulerSampler
from omegaconf import OmegaConf
from torch.utils.data.dataset import ConcatDataset
from torchviz import make_dot
import torch.optim as optim
import torch.multiprocessing as mp

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: HunYuanDiT,
        denoising_unet: HunYuanDiT,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
        ref_pose_guider=None,
        hand_depth_guider=None,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.ref_pose_guider = ref_pose_guider
        self.hand_depth_guider = hand_depth_guider

    def forward(
        self,
        x,
        t,
        ref_latents,
        tgt_pose_latents,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
        return_dict=True,
        controls=None,
        uncond_fwd: bool = False,
        ref_pose_img=None,
        hand_depth_img=None,
    ):
        pose_embedding = self.pose_guider(tgt_pose_latents)
        if self.ref_pose_guider:
            ref_pose_cond_tensor = ref_pose_img.to(device="cuda")
            ref_pose_fea = self.ref_pose_guider(ref_pose_cond_tensor)
            ref_pose_fea = ref_pose_fea.squeeze(dim=2)
        else:
            ref_pose_fea = None
        
        if self.hand_depth_guider:
            hand_depth_cond_tensor = hand_depth_img.to(device="cuda")
            hand_depth_fea = self.hand_depth_guider(hand_depth_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(t)
            self.reference_unet(
                ref_latents,
                ref_timesteps,
                encoder_hidden_states=encoder_hidden_states,
                text_embedding_mask=text_embedding_mask,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
                text_embedding_mask_t5=text_embedding_mask_t5,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=cos_cis_img,
                sin_cis_img=sin_cis_img,
                return_dict=return_dict,
                controls=controls,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            x,
            t,
            encoder_hidden_states=encoder_hidden_states,
            text_embedding_mask=text_embedding_mask,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
            text_embedding_mask_t5=text_embedding_mask_t5,
            image_meta_size=image_meta_size,
            style=style,
            cos_cis_img=cos_cis_img,
            sin_cis_img=sin_cis_img,
            return_dict=return_dict,
            controls=controls,
            pose_embedding=pose_embedding,
        )

        return model_pred

def print_tensor_info(tensor, tensor_name, file):
    file.write(f"Tensor name: {tensor_name}\n")
    file.write(f"Tensor: {tensor}\n")
    file.write(f"grad_fn: {tensor.grad_fn}\n")
    file.write("-" * 50 + "\n")

def deepspeed_initialize(args, logger, model, opt, deepspeed_config):
    logger.info(f"Initialize deepspeed...")
    logger.info(f"    Using deepspeed optimizer")

    def get_learning_rate_scheduler(warmup_min_lr, lr, warmup_num_steps, opt):
        return WarmupLR(opt, warmup_min_lr, lr, warmup_num_steps)

    logger.info(f"    Building scheduler with warmup_min_lr={args.warmup_min_lr}, warmup_num_steps={args.warmup_num_steps}")
    model, opt, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=get_trainable_params(model),
        config_params=deepspeed_config,
        args=args,
        lr_scheduler=partial(get_learning_rate_scheduler, args.warmup_min_lr, args.lr, args.warmup_num_steps) if args.warmup_num_steps > 0 else None,
    )
    return model, opt, scheduler

def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            if args.use_fp16:
                model.module.module.save_pretrained(cur_ckpt_save_dir)
            else:
                model.module.save_pretrained(cur_ckpt_save_dir)

    checkpoint_path = "[Not rank 0. Disabled output.]"

    client_state = {
        "steps": train_steps,
        "epoch": epoch,
        "args": args
    }
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    dst_paths = []
    if train_steps % args.ckpt_every == 0:
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt")
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=f"{train_steps:07d}.pt")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
        save_name = "latest.pt"
        checkpoint_path = f"{checkpoint_dir}/{save_name}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=f"{save_name}")
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=f"{save_name}")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    dist.barrier()
    if rank == 0 and len(dst_paths) > 0:
        # Delete optimizer states to avoid occupying too much disk space.
        for dst_path in dst_paths:
            for opt_state_path in glob(f"{dst_path}/zero_dp_rank_*_tp_rank_00_pp_rank_00_optim_states.pt"):
                os.remove(opt_state_path)

    return checkpoint_path

@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, freqs_cis_img):
    image = batch['img']
    tgt_pose = batch['tgt_pose']
    ref_image = batch['ref_img']
    # additional condition
    image_meta_size = batch['image_meta_size'].to(device)
    style = batch['style'].to(device)

    batch_size = image.shape[0]

    # encoder_hidden_states.shape torch.Size([1, 77, 1024])
    # encoder_hidden_states_t5.shape  torch.Size([1, 256, 2048])
    encoder_hidden_states = torch.load('empty_prompt_resources/encoder_hidden_states.pt').to(device)
    encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)
    encoder_hidden_states_t5 = torch.load('empty_prompt_resources/encoder_hidden_states_t5.pt').to(device)
    encoder_hidden_states_t5 = encoder_hidden_states_t5.repeat(batch_size, 1, 1)
    text_embedding_mask = torch.load('empty_prompt_resources/text_embedding_mask.pt').to(device)
    text_embedding_mask = text_embedding_mask.repeat(batch_size, 1)
    text_embedding_mask_t5 = torch.load('empty_prompt_resources/text_embedding_mask_t5.pt').to(device)
    text_embedding_mask_t5 = text_embedding_mask_t5.repeat(batch_size, 1)

    if args.extra_fp16:
        image = image.half()
        image_meta_size = image_meta_size.half() if image_meta_size is not None else None

    # Map input images to latent space + normalize latents:
    image = image.to(device)
    vae_scaling_factor = vae.config.scaling_factor
    latents = vae.encode(image).latent_dist.sample().mul_(vae_scaling_factor)
    ref_image = ref_image.to(device)
    ref_latents = vae.encode(ref_image).latent_dist.sample().mul_(vae_scaling_factor)
    tgt_pose = tgt_pose.to(device)
    tgt_pose_latents = vae.encode(tgt_pose).latent_dist.sample().mul_(vae_scaling_factor)

    # positional embedding
    _, _, height, width = image.shape
    reso = f"{height}x{width}"
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        image_meta_size=image_meta_size,
        style=style,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
        ref_latents=ref_latents,
        tgt_pose_latents=tgt_pose_latents,
    )

    return latents, model_kwargs

def main(args):
    if args.training_parts == "lora":
        args.use_ema = False

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    global_batch_size = world_size * batch_size * grad_accu_steps

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    deepspeed_config = deepspeed_config_from_args(args, global_batch_size)

    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    args_dict['world_size'] = world_size
    with open(f"{experiment_dir}/args.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Disable the message "Some weights of the model checkpoint at ... were not used when initializing BertModel."
    # If needed, just comment the following line.
    tf_logging.set_verbosity_error()

    # ===========================================================================
    # Building HYDIT
    # ===========================================================================

    logger.info("Building HYDIT Model.")

    # ---------------------------------------------------------------------------
    #   Training sample base size, such as 256/512/1024. Notice that this size is
    #   just a base size, not necessary the actual size of training samples. Actual
    #   size of the training samples are correlated with `resolutions` when enabling
    #   multi-resolution training.
    # ---------------------------------------------------------------------------
    image_size = args.image_size
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder). " \
                                                              f"got {image_size}"
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    # initialize model by deepspeed
    assert args.deepspeed, f"Must enable deepspeed in this script: train_deepspeed.py"

    model = HUNYUAN_DIT_MODELS[args.model](args,
                                    input_size=latent_size,
                                    log_fn=logger.info,
                                    )
    model_reference = HUNYUAN_DIT_MODELS[args.model](args,
                                    input_size=latent_size,
                                    log_fn=logger.info,
                                    )
    
    # Multi-resolution / Single-resolution training.
    if args.multireso:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      step=args.reso_step,
                                      target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      target_ratios=['1:1']).data

    freqs_cis_img = init_image_posemb(args.rope_img,
                                      resolutions=resolutions,
                                      patch_size=model.patch_size,
                                      hidden_size=model.hidden_size,
                                      num_heads=model.num_heads,
                                      log_fn=logger.info,
                                      rope_real=args.rope_real,
                                      )

    # Create EMA model and convert to fp16 if needed.

    ema, ema_reference = None, None
    if args.use_ema:
        ema = EMA(args, model, device, logger)
        ema_reference = EMA(args, model_reference, device, logger)

    # Setup FP16 main model:
    if args.use_fp16:
        model = Float16Module(model, args)
        model_reference = Float16Module(model_reference, args)
    logger.info(f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    # Setup VAE
    logger.info(f"    Loading vae from {VAE_EMA_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)

    if args.extra_fp16:
        logger.info(f"    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half().to(device)
    else:
        vae = vae.to(device)
    tokenizer = None
    tokenizer_t5 = None

    # setup pose guider
    pose_guider = PoseGuider()
    pose_guider_dict = torch.load('weights/pose_guider.pt')
    pose_guider.load_state_dict(pose_guider_dict)
    if args.use_fp16:
        pose_guider = Float16Module(pose_guider, args)

    logger.info(f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("    Using deepspeed optimizer")
    opt = None

    # ===========================================================================
    # Building Dataset
    # ===========================================================================

    logger.info(f"Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")

    cfg = OmegaConf.load(args.data_config)
    dataset1 = HumanAnimationImageDataset(data_config=cfg.data, img_size=(cfg.data.train_width, cfg.data.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
    dataset2 = HumanAnimationImageDataset(data_config=cfg.data2, img_size=(cfg.data2.train_width, cfg.data2.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
    dataset_list = [dataset1, dataset2]
    if 'data3' in cfg.keys():
        dataset3 = HumanAnimationImageDataset(data_config=cfg.data3, img_size=(cfg.data3.train_width, cfg.data3.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
        dataset_list.append(dataset3)
    if 'data4' in cfg.keys():
        dataset4 = HumanAnimationImageDataset(data_config=cfg.data4, img_size=(cfg.data4.train_width, cfg.data4.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
        dataset_list.append(dataset4)

    dataset = ConcatDataset(dataset_list)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=BatchSchedulerSampler(dataset, batch_size=batch_size), shuffle=False, num_workers=0
    )

    logger.info(f"    Dataset contains {len(dataset):,} images.")

    # ===========================================================================
    # Loading parameter
    # ===========================================================================
    logger.info(f"Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    # Resume checkpoint if needed
    if args.resume is not None or len(args.resume) > 0:
        model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))
        model_reference, ema_reference, start_epoch, start_epoch_step, train_steps = model_resume(args, model_reference, ema_reference, logger, len(loader))

    for name, param in model.named_parameters():
        if "module.blocks.39.attn2" in name or \
            'module.blocks.39.norm3' in name or \
                'module.blocks.39.norm2' in name or \
                    'module.blocks.39.mlp' in name or \
                        'module.final_layer' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        model_reference,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        model,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        model_reference,
        model,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    logger.info(f"    Training parts: {args.training_parts}")

    print_times = 'before_ds'
    print_file = open(f"tensor_info_{print_times}.txt", "w")
    for name, param in net.named_parameters():
        print_tensor_info(param, f"Model parameter {name}", print_file)
    print_file.close()

    # net, opt, scheduler = deepspeed_initialize(args, logger, net, opt, deepspeed_config)
    opt = optim.SGD(get_trainable_params(net), lr=0.001)

    print_times = 'after_ds'
    print_file = open(f"tensor_info_{print_times}.txt", "w")
    for name, param in net.named_parameters():
        print_tensor_info(param, f"Model parameter {name}", print_file)
    print_file.close()

    # ===========================================================================
    # Training
    # ===========================================================================

    net.train()
    if args.use_ema:
        ema.eval()

    print(f"    Worker {rank} ready.")
    dist.barrier()

    iters_per_epoch = len(loader)
    logger.info(" ****************************** Running training ******************************")
    logger.info(f"      Number GPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(f"      Number parameters:         {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(model)):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,} (world_size * batch_size * grad_accu_steps)")
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")

    logger.info(f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(f"      Training total steps:      {train_steps:,}/{min(args.max_training_steps, args.epochs * iters_per_epoch):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")

    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        logger.info(f"      Using EMA decay:           {ema.max_value if args.use_ema else None}")
        logger.info(f"      Using EMA warmup power:    {ema.power if args.use_ema else None}")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info("    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    if args.async_ema:
        ema_stream = torch.cuda.Stream()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"    Start random shuffle with seed={seed}")
        # Makesure all processors use the same seed to shuffle dataset.
        logger.info(f"    End of random shuffle")

        logger.info(f"    Beginning epoch {epoch}...")
        step = 0
        for batch in loader:
            step += 1

            import ipdb; ipdb.set_trace();
            latents, model_kwargs = prepare_model_inputs(args, batch, device, vae, freqs_cis_img)

            # training model by deepspeed while use fp16
            if args.use_fp16:
                if args.use_ema and args.async_ema:
                    with torch.cuda.stream(ema_stream):
                        ema.update(model.module.module, step=step)
                        ema_reference.update(model_reference.module.module, step=step)
                    torch.cuda.current_stream().wait_stream(ema_stream)

            loss_dict = diffusion.training_losses(model=net, x_start=latents, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            import ipdb;ipdb.set_trace();
            
            print_times = f'step_{step}'
            print_file = open(f"tensor_info_{print_times}.txt", "w")
            for name, param in net.named_parameters():
                print_tensor_info(param, f"Model parameter {name}", print_file)
            print_file.close()

            # net.backward(loss)
            loss.backward()
            last_batch_iteration = (train_steps + 1) // (global_batch_size // (batch_size * world_size))
            opt.step()
            # net.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})

            if args.use_ema and not args.async_ema or (args.async_ema and step == len(loader)-1):
                if args.use_fp16:
                    ema.update(model.module.module, step=step)
                    ema_reference.update(model_reference.module.module, step=step)
                else:
                    ema.update(model.module, step=step)
                    ema_reference.update(model_reference.module, step=step)

            # ===========================================================================
            # Log loss values:
            # ===========================================================================
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                # get lr from deepspeed fused optimizer
                logger.info(f"(step={train_steps:07d}) " +
                            (f"(update_step={train_steps // args.grad_accu_steps:07d}) " if args.grad_accu_steps > 1 else "") +
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Lr: {opt.param_groups[0]['lr']:.6g}, "
                            f"Steps/Sec: {steps_per_sec:.2f}, "
                            f"Samples/Sec: {int(steps_per_sec * batch_size * world_size):d}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # collect gc:
            if args.gc_interval > 0 and (step % args.gc_interval == 0):
                gc.collect()

            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0  # or train_steps == args.max_training_steps
                ) and train_steps > 0:
                save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir)

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking step loop at {train_steps}.")
                break

        if train_steps >= args.max_training_steps:
            logger.info(f"Breaking epoch loop at {epoch}.")
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    # Start
    main(get_args())
