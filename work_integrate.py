from PIL import Image
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image
from InstantCharacter.style_prompt import get_prompt
import time
pipe = None

path = "/workspaces/nikbauer34/tbank_imagegen/models/flux-4bit"
image_encoder_path = "/workspaces/nikbauer34/tbank_imagegen/models/openai"
ip_adapter_path = "/workspaces/nikbauer34/tbank_imagegen/models/xlabs-ip-adapter"
dtype = torch.bfloat16

def load_models():
    global pipe
    print("Loading transformer")
    start = time.time()
    model_4bit = FluxTransformer2DModel.from_pretrained(
        path, subfolder="transformer",torch_dtype=dtype
    )
    end = time.time()
    print(end - start)

    print("Loading text")
    start = time.time()
    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        path,
        subfolder="text_encoder_2",
        torch_dtype=dtype
    )
    end = time.time()
    print(end - start)
    print("Loading pipe")
    start = time.time()
    pipe = FluxPipeline.from_pretrained(
        path,
        transformer=model_4bit,
        text_encoder_2=text_encoder_2_4bit,
        torch_dtype=dtype,
        device_map="balanced",
    )
    end = time.time()
    print(end - start)
    
    print("Loading adapter")
    start = time.time()
    pipe.load_ip_adapter(
        ip_adapter_path,
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=image_encoder_path,
        torch_dtype=dtype,
    )
    end = time.time()
    print(end - start)
def generate_image_to_image(prompt: str, style_key: str, image: Image):
    global pipe
    # image = Image.open("/workspaces/nikbauer34/tbank_imagegen/input-ip-adapter.jpg")

    pipe.set_ip_adapter_scale(1.0)
    image = pipe(
        width=1024,
        height=1024,
        prompt=get_prompt(prompt, style_key),
        negative_prompt="",
        true_cfg_scale=4.0,
        generator=torch.Generator().manual_seed(4444),  # этот параметр лучше убрать
        ip_adapter_image=image,
    ).images[0]
    return image
def generate_image_from_request(prompt: str, style_key: str):
# сгружаем ip-adapter, затем заново загружаем
    global pipe
    pipe.unload_ip_adapter()
    image = pipe(
        width=1024,
        height=1024,
        prompt=get_prompt(prompt, style_key),
        negative_prompt="",
        true_cfg_scale=4.0,
        generator=torch.Generator().manual_seed(4444),  # этот параметр лучше убрать
    ).images[0]
    # загружаем обратно
    pipe.load_ip_adapter(
        ip_adapter_path,
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path=image_encoder_path,
        torch_dtype=dtype,
    )
    pipe.set_ip_adapter_scale(1.0)
    return image