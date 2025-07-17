import torch
import time
from diffusers import AutoModel
from transformers import T5EncoderModel


####### ЗАГРУЗКА TEXT 2 IMG МОДЕЛИ
path = "/workspaces/nikbauer34/tbank_imagegen/models/flux-8bit"
print("Loading")
start = time.time()
model_8bit = AutoModel.from_pretrained(
    path, subfolder="transformer",torch_dtype=torch.bfloat16
)
end = time.time()
print(end - start)

path = "/workspaces/nikbauer34/tbank_imagegen/models/flux-8bit"
print("Loading text")
start = time.time()
text_encoder_2_8bit = T5EncoderModel.from_pretrained(
    path,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16
)
end = time.time()
print(end - start)

from diffusers import FluxPipeline

print("Loading full")
start = time.time()
pipe = FluxPipeline.from_pretrained(
    path,
    transformer=model_8bit,
    text_encoder_2=text_encoder_2_8bit,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
end = time.time()
print(end - start)
###########################################################################


############ Пример
pipe_kwargs = {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 20,
    "max_sequence_length": 512,
}

image = pipe(**pipe_kwargs, generator=torch.manual_seed(42),).images[0]
image.save("test1.jpg")




###############################################################
# 1. FastApi server с text2img ручкой
