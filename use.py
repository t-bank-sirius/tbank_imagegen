# from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
# from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
# import torch
# from diffusers import FluxTransformer2DModel, FluxPipeline
# from transformers import T5EncoderModel
# from diffusers.utils import load_image
# from diffusers import AutoModel, FluxPipeline

# model_path="/workspaces/nikbauer34/tbank_imagegen/models/flux-8bit"
# print("Loading transformer")
# model_8bit = AutoModel.from_pretrained(
#     model_path, subfolder="transformer", torch_dtype=torch.bfloat16
# )
# print("Loading text_encoder_2")
# text_encoder_2_8bit = T5EncoderModel.from_pretrained(
#     model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
# )
# pipe = FluxPipeline.from_pretrained(
#     "/workspaces/nikbauer34/tbank_imagegen/models/flux",
#     transformer=model_8bit,
#     text_encoder_2=text_encoder_2_8bit,
#     torch_dtype=torch.float16,
# )

# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux_ip_adapter_input.jpg"
# ).resize((1024, 1024))

# pipe.load_ip_adapter(
#     "XLabs-AI/flux-ip-adapter",
#     weight_name="ip_adapter.safetensors",
#     image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
#     torch_dtype=torch.float16,
# )

# pipe.enable_model_cpu_offload()

# pipe.set_ip_adapter_scale(1.0)

# image = pipe(
#     width=1024,
#     height=1024,
#     prompt="wearing sunglasses",
#     negative_prompt="",
#     true_cfg_scale=4.0,
#     generator=torch.Generator().manual_seed(4444),
#     ip_adapter_image=image,
# ).images[0]

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image


path = "/workspaces/nikbauer34/tbank_imagegen/models/flux"
save_path = "/workspaces/nikbauer34/tbank_imagegen/models/flux-4bit"
dtype = torch.bfloat16


quant_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

print("T5")
text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    path,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=dtype,
)
print("T5")

quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

print("Transformer")
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    path,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=dtype,
)
print("Transformer")

pipe = FluxPipeline.from_pretrained(
    path,
    transformer=transformer_4bit,
    text_encoder_2=text_encoder_2_4bit,
    torch_dtype=dtype,
    # device_map="balanced",
)
print("Done")
pipe.save_pretrained(save_directory=save_path)


image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux_ip_adapter_input.jpg"
).resize((1024, 1024))

image_encoder_path = "/workspaces/nikbauer34/tbank_imagegen/models/openai"
ip_adapter_path = "/workspaces/nikbauer34/tbank_imagegen/models/xlabs-ip-adapter"
pipe.load_ip_adapter(
    ip_adapter_path,
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path=image_encoder_path,
    torch_dtype=dtype,
)

pipe.enable_model_cpu_offload()

pipe.set_ip_adapter_scale(1.0)

image = pipe(
    width=1024,
    height=1024,
    prompt="wearing sunglasses",
    negative_prompt="",
    true_cfg_scale=4.0,
    generator=torch.Generator().manual_seed(4444),
    ip_adapter_image=image,
).images[0]

image.save("flux_ip_adapter_output.jpg")
