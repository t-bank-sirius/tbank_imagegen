
from huggingface_hub import login
import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

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

# Применяем квантование на 8 бит (qfloat8) и замораживаем веса
quantize(transformer, weights=qfloat8)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,  # будет добавлено ниже
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,     # будет добавлено ниже
)

# Присваиваем квантованные модели пайплайну
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()

generator = torch.Generator().manual_seed(12345)
image = pipe(
    prompt='a black cat from Greek period',
    width=1024,
    height=1024,
    num_inference_steps=4,
    generator=generator,
    guidance_scale=3.5,
).images[0]
image.save('test_flux_distilled_1.png')