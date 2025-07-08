import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pipeline import InstantCharacterFluxPipeline

import io

# ==== MODEL PRELOAD ====
ip_adapter_path = 'checkpoints/instantcharacter_ip-adapter.bin'
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
seed = 123456

pipe = InstantCharacterFluxPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
)
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)
pipe.enable_sequential_cpu_offload()

app = FastAPI()

ref_image_path = 'boy.jpg'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

# Step 3 Inference without style
prompt = "This exact boy in the anime world"
image = pipe(
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    generator=torch.manual_seed(seed),
).images[0]
image.save("flux.png")