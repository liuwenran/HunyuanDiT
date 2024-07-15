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
from hydit.utils.tools import create_logger, set_seeds, create_exp_folder, model_resume, get_trainable_params, model_copy_to_reference_net
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
import shutil
from PIL import Image
from transformers import CLIPVisionModelWithProjection

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: HunYuanDiT,
        denoising_unet: HunYuanDiT,
        pose_guider: PoseGuider,
        reference_control_writer=None,
        reference_control_reader=None,
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
        clip_img_embedding=None,
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
            ref_pred = self.reference_unet(
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
                clip_img_embedding=clip_img_embedding,
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
            clip_img_embedding=clip_img_embedding,
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
        folders = [folder for folder in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, folder))]
        folders_to_delete = [folder for folder in folders if "latest" not in folder]
        sorted_folders = sorted(folders_to_delete, key=lambda x: int(x.split(".")[0]))
        for folder in sorted_folders[:-1]:
            folder_path = os.path.join(checkpoint_dir, folder)
            shutil.rmtree(folder_path)
        # Delete optimizer states to avoid occupying too much disk space.
        for dst_path in dst_paths:
            for opt_state_path in glob(f"{dst_path}/zero_pp_rank_*_mp_rank_00_optim_states.pt"):
                os.remove(opt_state_path)

    return checkpoint_path

@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, freqs_cis_img, encoder_hidden_states, encoder_hidden_states_t5, text_embedding_mask, text_embedding_mask_t5):
    image = batch['img']
    tgt_pose = batch['tgt_pose']
    ref_image = batch['ref_img']
    # additional condition
    image_meta_size = batch['image_meta_size'].to(device)
    style = batch['style'].to(device)

    batch_size = image.shape[0]

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


def validation(args, model, model_reference, pose_guider, vae, target_width, target_height, freqs_cis_img, val_save_dir, global_step):
    from hydit.inference_human_animation import get_pipeline
    pipeline, sampler = get_pipeline(args, vae, None, None, model, model_reference, pose_guider, model.device, 0, None, 'torch')

    generator = set_seeds(2024, device=model.device)

    size_cond = [target_width, target_height, target_width, target_height, 0, 0]
    image_meta_size = torch.as_tensor([size_cond] * 2 , device=model.device)

    reso = f"{target_height}x{target_width}"
    freqs_cis_img = freqs_cis_img[reso]

    ref_image_paths = open(args.validation_ref_images, 'r').readlines()
    tgt_image_paths = open(args.validation_tgt_images, 'r').readlines()

    pil_images = []
    for ind in range(len(ref_image_paths)):
        for tgt_ind in range(len(tgt_image_paths)):

            ref_image_path = ref_image_paths[ind].strip()
            tgt_image_path = tgt_image_paths[tgt_ind].strip()

            ref_name = ref_image_path.split("/")[-1].split('.')[0]
            tgt_name = tgt_image_path.split("/")[-1].split('.')[0]
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            tgt_image_pil = Image.open(tgt_image_path).convert("RGB")

            control_image = tgt_image_pil
            
            samples = pipeline(
                ref_image_pil,
                tgt_image_pil,
                height=target_height,
                width=target_width,
                prompt='',
                negative_prompt='',
                num_images_per_prompt=1,
                guidance_scale=6.0,
                num_inference_steps=100,
                image_meta_size=image_meta_size,
                style=torch.as_tensor([0, 0], device=model.device),
                return_dict=False,
                generator=generator,
                freqs_cis_img=freqs_cis_img,
                use_fp16=True,
                learn_sigma=True,
            )[0]

            res_image_pil = samples[0]

            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil.size

            ref_image_pil = ref_image_pil.resize((w, h))
            control_image = control_image.resize((w, h))
            canvas = Image.new("RGB", (w * 3, h), "white")

            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(control_image, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            sample_name = f"{ref_name}_{tgt_name}"
            out_file = Path(f"{val_save_dir}/{global_step:06d}-{sample_name}.png")
            canvas.save(out_file)

    del pipeline
    torch.cuda.empty_cache()

    return pil_images

def main(args):
    if args.training_parts == "lora":
        args.use_ema = False

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # dist.init_process_group("nccl")
    deepspeed.init_distributed()

    world_size = dist.get_world_size()
    print('world_size')
    print(world_size)
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
    if rank == 0:
        print('show args')
        print(args)
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    args_dict['world_size'] = world_size
    if rank == 0:
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
    model = HUNYUAN_DIT_MODELS[args.model](args,
                                    input_size=latent_size,
                                    log_fn=logger.info,
                                    )

    # model_setting = 'DiT-g/2-ref10'
    model_setting = args.model
    model_reference = HUNYUAN_DIT_MODELS[model_setting](args,
                                        input_size=latent_size,
                                        log_fn=logger.info,
                                        )
    
    pose_guider = PoseGuider()
    
    hidden_size = model.hidden_size
    num_tokens = model.x_embedder.num_patches

    reference_control_writer = ReferenceAttentionControl(
        model_reference,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_tokens=num_tokens,
    )
    reference_control_reader = ReferenceAttentionControl(
        model,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_tokens=num_tokens,
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_base_model_path,
        subfolder="image_encoder",
    ).to(device=device)

    assert args.deepspeed, f"Must enable deepspeed in this script: train_deepspeed.py"
    with deepspeed.zero.Init(data_parallel_group=torch.distributed.group.WORLD,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=deepspeed_config,
                             mpu=None,
                             enabled=args.zero_stage == 3):

        net = Net(
            model_reference,
            model,
            pose_guider,
            reference_control_writer,
            reference_control_reader,
        )

    # Multi-resolution / Single-resolution training.
    base_size = (image_size[0] + image_size[1]) // 2
    if args.multireso:
        resolutions = ResolutionGroup(base_size,
                                      align=16,
                                      step=args.reso_step,
                                      target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(base_size,
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
    dataset_list = [dataset1]
    if 'data2' in cfg.keys():
        dataset2 = HumanAnimationImageDataset(data_config=cfg.data2, img_size=(cfg.data2.train_width, cfg.data2.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
        dataset_list.append(dataset2)
    if 'data3' in cfg.keys():
        dataset3 = HumanAnimationImageDataset(data_config=cfg.data3, img_size=(cfg.data3.train_width, cfg.data3.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
        dataset_list.append(dataset3)
    if 'data4' in cfg.keys():
        dataset4 = HumanAnimationImageDataset(data_config=cfg.data4, img_size=(cfg.data4.train_width, cfg.data4.train_height), control_type=cfg.control_type, use_depth_enhance=cfg.use_depth_enhance, use_ref_pose_guider=cfg.use_ref_pose_guider, use_hand_depth=cfg.use_hand_depth)
        dataset_list.append(dataset4)

    dataset = ConcatDataset(dataset_list)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=BatchSchedulerSampler(dataset, batch_size=batch_size), shuffle=False, num_workers=4
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
    logger.info(" ****************************** Loading parameter ******************************")
    args.strict = False
    if args.resume is not None or len(args.resume) > 0:
        resume_ckpt_module = torch.load('/mnt/petrelfs/liuwenran/forks/HunyuanDiT/weights/hunyuandit_converted.pth',
                                        map_location=lambda storage, loc: storage)
        state_dict_to_load = {}
        for key in resume_ckpt_module.keys():
            # if 'attn2.kv_proj.weight' in key:
            if 'bank' in key:
                pass
            else:
                state_dict_to_load[key] = resume_ckpt_module[key]
        model.load_state_dict(state_dict_to_load, strict=args.strict)
        # model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))

    model, model_reference = model_copy_to_reference_net(model, model_reference)

    # # setup pose guider
    pose_guider_dict = torch.load('weights/pose_guider.pt', map_location=lambda storage, loc: storage)
    pose_guider.load_state_dict(pose_guider_dict)
    del pose_guider_dict

    if args.use_fp16:
        net = Float16Module(net, args)

    # model_reference_block_end_ind = len(model_reference.blocks)
    # grad_no_need_layers = [f'blocks.{model_reference_block_end_ind}.attn2', 
    #                        f'blocks.{model_reference_block_end_ind}.norm3',
    #                        f'blocks.{model_reference_block_end_ind}.norm2',
    #                        f'blocks.{model_reference_block_end_ind}.mlp']
    # only ban final layer for simple code
    for name, param in model_reference.named_parameters():
        if 'final_layer' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    logger.info(f"    Training parts: {args.training_parts}")

    net, opt, scheduler = deepspeed_initialize(args, logger, net, opt, deepspeed_config)

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

    # encoder_hidden_states.shape torch.Size([1, 77, 1024])
    # encoder_hidden_states_t5.shape  torch.Size([1, 256, 2048])
    encoder_hidden_states_empty = torch.load('prompt_embeddings/empty_prompt_train/encoder_hidden_states.pt', map_location="cpu").to(device)
    encoder_hidden_states_empty = encoder_hidden_states_empty.repeat(batch_size, 1, 1)
    encoder_hidden_states_t5_empty = torch.load('prompt_embeddings/empty_prompt_train/encoder_hidden_states_t5.pt', map_location="cpu").to(device)
    encoder_hidden_states_t5_empty = encoder_hidden_states_t5_empty.repeat(batch_size, 1, 1)
    text_embedding_mask_empty = torch.load('prompt_embeddings/empty_prompt_train/text_embedding_mask.pt', map_location="cpu").to(device)
    text_embedding_mask_empty = text_embedding_mask_empty.repeat(batch_size, 1)
    text_embedding_mask_t5_empty = torch.load('prompt_embeddings/empty_prompt_train/text_embedding_mask_t5.pt', map_location="cpu").to(device)
    text_embedding_mask_t5_empty = text_embedding_mask_t5_empty.repeat(batch_size, 1)

    encoder_hidden_states_human = torch.load('prompt_embeddings/human_prompt_train/encoder_hidden_states.pt', map_location="cpu").to(device)
    encoder_hidden_states_human = encoder_hidden_states_human.repeat(batch_size, 1, 1)
    encoder_hidden_states_t5_human = torch.load('prompt_embeddings/human_prompt_train/encoder_hidden_states_t5.pt', map_location="cpu").to(device)
    encoder_hidden_states_t5_human = encoder_hidden_states_t5_human.repeat(batch_size, 1, 1)
    text_embedding_mask_human = torch.load('prompt_embeddings/human_prompt_train/text_embedding_mask.pt', map_location="cpu").to(device)
    text_embedding_mask_human = text_embedding_mask_human.repeat(batch_size, 1)
    text_embedding_mask_t5_human = torch.load('prompt_embeddings/human_prompt_train/text_embedding_mask_t5.pt', map_location="cpu").to(device)
    text_embedding_mask_t5_human = text_embedding_mask_t5_human.repeat(batch_size, 1)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"    Start random shuffle with seed={seed}")
        # Makesure all processors use the same seed to shuffle dataset.
        logger.info(f"    End of random shuffle")

        logger.info(f"    Beginning epoch {epoch}...")
        step = 0
        for batch in loader:
            step += 1

            if random.random() < args.uncond_p:
                encoder_hidden_states = encoder_hidden_states_empty
                encoder_hidden_states_t5 = encoder_hidden_states_t5_empty
                text_embedding_mask = text_embedding_mask_empty
                text_embedding_mask_t5 = text_embedding_mask_t5_empty
            else:
                encoder_hidden_states = encoder_hidden_states_human
                encoder_hidden_states_t5 = encoder_hidden_states_t5_human
                text_embedding_mask = text_embedding_mask_human
                text_embedding_mask_t5 = text_embedding_mask_t5_human

            latents, model_kwargs = prepare_model_inputs(args, batch, device, vae, freqs_cis_img, encoder_hidden_states, encoder_hidden_states_t5, text_embedding_mask, text_embedding_mask_t5)

            # training model by deepspeed while use fp16
            if args.use_fp16:
                if args.use_ema and args.async_ema:
                    with torch.cuda.stream(ema_stream):
                        ema.update(model.module.module, step=step)
                        ema_reference.update(model_reference.module.module, step=step)
                    torch.cuda.current_stream().wait_stream(ema_stream)

            loss_dict = diffusion.training_losses(model=net, x_start=latents, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            
            # if step == 1 or step % 10 == 0:
            #     print_times = f'step_{step}'
            #     print_file = open(f"tensor_info_denoising_{print_times}.txt", "w")
            #     for name, param in net.named_parameters():
            #         print_tensor_info(param, f"Model parameter {name}", print_file)
            #     print_file.close()

            net.backward(loss)
            last_batch_iteration = (train_steps + 1) // (global_batch_size // (batch_size * world_size))
            net.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})

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

            reference_control_reader.clear()
            reference_control_writer.clear()

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
            
            if train_steps % args.val_every == 0:
                dist.barrier()
                if rank == 0:
                    target_width, target_height = cfg.data.train_width, cfg.data.train_height
                    val_save_dir = f'{experiment_dir}/val'
                    os.makedirs(val_save_dir, exist_ok=True)
                    validation(args, model, model_reference, pose_guider, vae, target_width, target_height, freqs_cis_img, val_save_dir, train_steps)

            # collect gc:
            if args.gc_interval > 0 and (step % args.gc_interval == 0):
                gc.collect()

            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0  # or train_steps == args.max_training_steps
                ) and train_steps > 0:
                save_checkpoint(args, rank, logger, net, ema, epoch, train_steps, checkpoint_dir)

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
