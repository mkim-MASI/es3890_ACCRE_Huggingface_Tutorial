#https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
from diffusers import DiffusionPipeline
import torch
import os

output_dir="/home/kimm58/saved_img/"

#load the base model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe.to("cuda")

# Generate an image with the base model
prompt = "Sandcat rolling on the ground having fun with tongue out"
image = pipe(prompt=prompt).images[0]

# Save the generated image
output_path = os.path.join(output_dir, "testimg.png")
image.save(output_path)
