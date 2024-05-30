import torch
import gc
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def generate_custom_image():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    term = input("ENTER THE IMAGE THEME IN 1 OR 2 WORDS: ")
    prompt = f"a ((full-body:2)) shot of a ((single:2)) {text}, isolated on {bkgd_color} background, 4k, highly detailed"

    image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    image.save("output.png")

    from PIL import Image
    import matplotlib.pyplot as plt
    plt.imshow(Image.open("output.png"))
    plt.axis("off")
    plt.show()

    del unet, pipe
    torch.cuda.empty_cache()
    gc.collect()

generate_custom_image()
