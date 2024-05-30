import torch
import numpy as np
import rembg
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from src.utils.infer_util import remove_background, resize_foreground

import os
os.chdir('InstantMesh')

model = None
torch.cuda.empty_cache()

# Load and set up the pipeline
pipeline = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus", torch_dtype=torch.float16)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)
device = torch.device('cuda')
pipeline = pipeline.to(device)
seed_everything(0)

def preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image

def generate_mvs(input_image, sample_steps, sample_seed):
    seed_everything(sample_seed)
    generator = torch.Generator(device=device)
    z123_image = pipeline(
        input_image,
        num_inference_steps=sample_steps,
        generator=generator,
    ).images[0]
    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, 'h w c -> h w c')
    show_image = Image.fromarray(show_image.numpy())
    return z123_image, show_image

input_image_path = 'Output.png'  # Adjust the path as needed
input_image = Image.open(input_image_path)
processed_image = preprocess(input_image, True)
processed_image
mv_images, mv_show_images = generate_mvs(processed_image, 75, 42)
mv_images.save('InstantMesh/mv_images.png')
mv_show_images.show()
