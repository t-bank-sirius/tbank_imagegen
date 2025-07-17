import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from InstantCharacter.pipeline import InstantCharacterFluxPipeline
from huggingface_hub import login
from InstantCharacter.style_prompt import get_prompt
# login(token="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")
import io
pipe = None

def preload_model_image_to_image():
    """
    Инициализация пайпа и адаптера. Глобальная переменная _pipe.
    """
    global pipe
    import torch
    from InstantCharacter.pipeline import InstantCharacterFluxPipeline
    ip_adapter_path = '/workspaces/nikbauer34/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
    base_model = '/workspaces/nikbauer34/tbank_imagegen/models/flux'
    image_encoder_path = '/workspaces/nikbauer34/tbank_imagegen/models/google'
    image_encoder_2_path = '/workspaces/nikbauer34/tbank_imagegen/models/facebook'

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

def generate_image_to_image(image_prompt_list, seed=123456):
    """
    Принимает список словарей {'image': PIL.Image, 'prompt': str},
    возвращает список PIL.Image.
    Использует глобальный _pipe.
    """
    global pipe
    import torch

    if pipe is None:
        raise RuntimeError('Pipe не инициализирован. Вызовите init_pipe() сначала.')

    results = []
    for image_prompt in image_prompt_list:
        input_image = image_prompt['image']
        input_prompt = image_prompt['prompt']
        input_style = image_prompt['style_key']
        output_image = pipe(
            prompt=get_prompt(input_prompt, input_style),
            num_inference_steps=28,
            guidance_scale=3.5,
            subject_image=input_image,
            subject_scale=0.9,
            generator=torch.manual_seed(seed),
        ).images[0]
        results.append(output_image)
    return results
