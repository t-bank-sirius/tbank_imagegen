from huggingface_hub import login
import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
pipe = None  # Глобальная переменная пайплайна

def preload_model_prompt_to_image():
    global pipe
    login(token="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")
    dtype = torch.bfloat16

    bfl_repo = "black-forest-labs/FLUX.1-schnell"
    revision = "refs/pr/1"

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        bfl_repo, subfolder="scheduler", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        bfl_repo, subfolder="tokenizer_2", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision
    )

    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    pipe_local = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe_local.text_encoder_2 = text_encoder_2
    pipe_local.transformer = transformer
    pipe_local.enable_model_cpu_offload()
    pipe = pipe_local
    # return pipe

def generate_images_from_prompts(prompts):
    """
    prompts: список строк (текстовых промптов)
    Возвращает: список PIL.Image той же длины
    """
    global pipe
    if pipe is None:
        raise ValueError("Model is not loaded. Call preload_model() before inference.")
    generator = torch.Generator().manual_seed(12345)
    images = []
    for prompt in prompts:
        image = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=4,
            generator=generator,
            guidance_scale=3.5,
        ).images[0]
        images.append(image)
    return images