import torch
from PIL import Image
from InstantCharacter.pipeline import InstantCharacterFluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import (
    CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
# from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import FluxPipeline
from InstantCharacter.style_prompt import get_prompt

# --- Квантование (замените на ваши методы, если они отличаются) ---
from optimum.quanto import freeze, qfloat8, quantize  # настройте под свои функции и qfloat8/qint8

# --- Глобальные переменные ---
pipe_txt2img = None
pipe_imgtxt2img = None

def preload_model():
    """
    Инициализация и квантизация обеих моделей.
    """
    global pipe_txt2img, pipe_imgtxt2img

    # ----- TEXT-TO-IMAGE PIPELINE -----
    dtype = torch.bfloat16
    bfl_repo = "/workspaces/nikbauer34/tbank_imagegen/models/flux"
    openai_path = "/workspaces/nikbauer34/tbank_imagegen/models/openai"
    revision = "refs/pr/1"
    quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        bfl_repo, subfolder="scheduler", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        openai_path,  torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(openai_path)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo, quantization_config=quant_config, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        bfl_repo, subfolder="tokenizer_2", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision
    )
    quant_config_2 = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo, quantization_config=quant_config_2, subfolder="transformer", torch_dtype=dtype, revision=revision
    )

    # --- Квантизация ---
    # quantize(transformer, weights=qfloat8)
    # freeze(transformer)
    # quantize(text_encoder_2, weights=qfloat8)
    # freeze(text_encoder_2)

    # --- Init text-to-image pipeline ---
    pipe_txt2img_local = FluxPipeline.from_pretrained(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None
    )
    pipe_txt2img_local.text_encoder_2 = text_encoder_2
    pipe_txt2img_local.transformer = transformer
    pipe_txt2img_local.enable_model_cpu_offload()
    pipe_txt2img = pipe_txt2img_local

    # ----- IMAGE+TEXT-TO-IMAGE (INSTANT CHARACTER) PIPELINE -----
    ip_adapter_path = '/workspaces/nikbauer34/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
    image_encoder_path = '/workspaces/nikbauer34/tbank_imagegen/models/google'
    image_encoder_2_path = '/workspaces/nikbauer34/tbank_imagegen/models/facebook'

    pipe_imgtxt2img = InstantCharacterFluxPipeline.from_pretrained(
        scheduler=pipe_txt2img_local.scheduler,
        text_encoder=pipe_txt2img_local.text_encoder,
        tokenizer=pipe_txt2img_local.tokenizer,
        text_encoder_2=None,
        tokenizer_2=pipe_txt2img_local.tokenizer_2,
        vae=pipe_txt2img_local.vae,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )
    pipe_imgtxt2img.text_encoder_2 = pipe_txt2img_local.text_encoder_2
    pipe_txt2img_local.transformer = pipe_txt2img_local.transformer
    pipe_imgtxt2img.init_adapter(
        image_encoder_path=image_encoder_path,
        image_encoder_2_path=image_encoder_2_path,
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=512),  # Меньше токенов — легче адаптер
    )
    pipe_imgtxt2img.enable_sequential_cpu_offload()

    # --- Устанавливаем глобальные переменные ---
    globals()['pipe_txt2img'] = pipe_txt2img
    globals()['pipe_imgtxt2img'] = pipe_imgtxt2img

# --- Функция генерации для text-to-image ---
def generate_images_from_prompts(prompts):
    """
    prompts: список объектов с полями prompt и style_key
    Возвращает: список PIL.Image
    """
    global pipe_txt2img
    if pipe_txt2img is None:
        raise RuntimeError("Text-to-image модель не инициализирована.")

    generator = torch.Generator().manual_seed(12345)
    images = []
    for prompt in prompts:
        image = pipe_txt2img(
prompt=get_prompt(prompt.prompt, prompt.style_key),
            width=1024,
            height=1024,
            num_inference_steps=4,
            generator=generator,
            guidance_scale=3.5,
        ).images[0]
        images.append(image)
    return images

# --- Функция генерации для image+text-to-image ---
def generate_images_from_image_and_prompt(image_prompt_list, seed=123456):
    """
    image_prompt_list: список словарей {'image': PIL.Image, 'prompt': str, 'style_key': str}
    Возвращает: список PIL.Image
    """
    global pipe_imgtxt2img
    if pipe_imgtxt2img is None:
        raise RuntimeError("Image-text-to-image pipe не инициализирован.")

    results = []
    for image_prompt in image_prompt_list:
        input_image = image_prompt['image']
        input_prompt = image_prompt['prompt']
        input_style = image_prompt['style_key']
        output_image = pipe_imgtxt2img(
            prompt=get_prompt(input_prompt, input_style),
            num_inference_steps=28,
            guidance_scale=3.5,
            subject_image=input_image,
            subject_scale=0.9,
            generator=torch.manual_seed(seed),
        ).images[0]
        results.append(output_image)
    return results