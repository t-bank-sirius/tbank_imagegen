from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import torch
from diffusers import AutoModel
from transformers import T5EncoderModel
from pipeline import InstantCharacterFluxPipeline

quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True,)
path = "/workspaces/nikbauer34/tbank_imagegen/models/flux"
save_path = "/workspaces/nikbauer34/tbank_imagegen/models/flux-4bit"

text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    path,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
print("Text Encoder Loaded")


quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True,)
transformer_4bit = AutoModel.from_pretrained(
    path,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
print("Transformer Loaded")

pipe = InstantCharacterFluxPipeline.from_pretrained(
    path,
    transformer=transformer_4bit,
    text_encoder_2=text_encoder_2_4bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)
print("Pipe Done.")

pipe.save_pretrained(save_directory=save_path)
print("saving done")
