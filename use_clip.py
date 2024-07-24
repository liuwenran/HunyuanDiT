from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

# model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
# processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# model = CLIPVisionModelWithProjection.from_pretrained("/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189")
# processor = AutoProcessor.from_pretrained("/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189")

model = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

origin_processor = CLIPImageProcessor()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

image_embeds = outputs.image_embeds

origin_inputs = origin_processor(images=image, return_tensors="pt")

origin_outputs = model(**origin_inputs)

origin_image_embeds = origin_outputs.image_embeds

import ipdb;ipdb.set_trace();