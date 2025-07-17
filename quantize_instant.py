import torch
from transformers import AutoModel, SiglipVisionModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
# ckpt = "/workspaces/nikbauer34/tbank_imagegen/models/google"
# print("working")
# quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
# print("working 2")
# model = SiglipVisionModel.from_pretrained(ckpt, quantization_config=quant_config,  torch_dtype=torch.float16,)
# model.save_pretrained(save_directory="/workspaces/nikbauer34/tbank_imagegen/models/google-8bit")

# print("saved google")

ckpt = "/workspaces/nikbauer34/tbank_imagegen/models/facebook"
quant_config_2 = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
print("working 3")
model_2 = AutoModel.from_pretrained(ckpt, quantization_config=quant_config_2,  torch_dtype=torch.float16,)
model_2.save_pretrained(save_directory="/workspaces/nikbauer34/tbank_imagegen/models/facebook-8bit")