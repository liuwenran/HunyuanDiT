from pathlib import Path

from loguru import logger

from hydit.config import get_args
from hydit.inference_human_animation import End2End
from omegaconf import OmegaConf
from PIL import Image

def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    enhancer = None

    return args, gen, enhancer


if __name__ == "__main__":
    args, gen, enhancer = inferencer()

    infer_data_cfg = '/mnt/petrelfs/liuwenran/forks/HunyuanDiT/dataset/human_animation_yamls/stage1_infer.yaml'
    infer_data_cfg = OmegaConf.load(infer_data_cfg)

    enhanced_prompt = None

    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size
    for ref_image_path in infer_data_cfg.reference_image_path:
        for pose_image_path in infer_data_cfg.pose_image_path:
            ref_image = Image.open(ref_image_path).convert("RGB")
            pose_image = Image.open(pose_image_path).convert("RGB")
            results = gen.predict(ref_image,
                                pose_image,
                                args.prompt,
                                height=height,
                                width=width,
                                seed=args.seed,
                                enhanced_prompt=enhanced_prompt,
                                negative_prompt=args.negative,
                                infer_steps=args.infer_steps,
                                guidance_scale=args.cfg_scale,
                                batch_size=args.batch_size,
                                src_size_cond=args.size_cond,
                                )
            images = results['images']

            # Save images
            save_dir = Path('results')
            save_dir.mkdir(exist_ok=True)
            # Find the first available index
            all_files = list(save_dir.glob('*.png'))
            if all_files:
                start = max([int(f.stem) for f in all_files]) + 1
            else:
                start = 0

            for idx, pil_img in enumerate(images):
                save_path = save_dir / f"{idx + start}.png"
                pil_img.save(save_path)
                logger.info(f"Save to {save_path}")
