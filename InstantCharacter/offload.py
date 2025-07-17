import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
# Step 1 Load base model and adapter

ip_adapter_path = '/workspaces/nikbauer34/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
base_model = '/workspaces/nikbauer34/tbank_imagegen/models/flux'
image_encoder_path = '/workspaces/nikbauer34/tbank_imagegen/models/google'
image_encoder_2_path = '/workspaces/nikbauer34/tbank_imagegen/models/facebook'
seed = 123456
dtype=torch.bfloat16

pipe = InstantCharacterFluxPipeline.from_pretrained("/workspaces/nikbauer34/tbank_imagegen/models/flux-instant-3",  
    
    torch_dtype=dtype,)

# Step 1.1, To manually configure the CPU offload mode.
# You may selectively designate which layers to employ the offload hook based on the available VRAM capacity of your GPU.
# The following configuration can reach about 22GB of VRAM usage on NVIDIA L20 (Ada arch)

# pipe.to("cpu")
pipe._exclude_from_cpu_offload.extend([
    # 'vae',
    'text_encoder',
    # 'text_encoder_2',
])
pipe._exclude_layer_from_cpu_offload.extend([
    "transformer.pos_embed",
    "transformer.time_text_embed",
    "transformer.context_embedder",
    "transformer.x_embedder",
    "transformer.transformer_blocks",
    # "transformer.single_transformer_blocks",
    "transformer.norm_out",
    "transformer.proj_out",
])
pipe.reset_device_map()
pipe.enable_sequential_cpu_offload()

pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
    device=torch.device('cuda')
)

# Step 1.2 Optional inference acceleration
# You can set the TORCHINDUCTOR_CACHE_DIR in production environment.

# torch._dynamo.reset()
# torch._dynamo.config.cache_size_limit = 1024
# torch.set_float32_matmul_precision("high")
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# for layer in pipe.transformer.attn_processors.values():
#     layer = torch.compile(
#         layer,
#         fullgraph=True,
#         dynamic=True,
#         mode="max-autotune",
#         backend='inductor'
#     )
# pipe.transformer.single_transformer_blocks.compile(
#     fullgraph=True,
#     dynamic=True,
#     mode="max-autotune",
#     backend='inductor'
# )
# pipe.transformer.transformer_blocks.compile(
#     fullgraph=True,
#     dynamic=True,
#     mode="max-autotune",
#     backend='inductor'
# )
# pipe.vae = torch.compile(
#     pipe.vae,
#     fullgraph=True,
#     dynamic=True,
#     mode="max-autotune",
#     backend='inductor'
# )
# pipe.text_encoder = torch.compile(
#     pipe.text_encoder,
#     fullgraph=True,
#     dynamic=True,
#     mode="max-autotune",
#     backend='inductor'
# )


# Step 2 Load reference image
ref_image_path = 'assets/girl.jpg'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')

ref_image_path_2 = 'assets/boy.jpg'  # white background
ref_image_2 = Image.open(ref_image_path_2).convert('RGB')
# Step 3 Inference without style
prompt = "there are 2 separate pictures that are split horizontally. You will need to create one picture based on this prompt: move both of them outside and make a picture with both of them"
img1 = ref_image
img2 = ref_image_2
# Горизонтально
width = img1.width + img2.width
height = max(img1.height, img2.height)
concat = Image.new('RGB', (width, height))
concat.paste(img1, (0, 0))
concat.paste(img2, (img1.width, 0))

# warm up for torch.compile
# image = pipe(
#         prompt=prompt, 
#         num_inference_steps=28,
#         guidance_scale=3.5,
#         subject_image=ref_image,
#         subject_scale=0.9,
#         generator=torch.manual_seed(seed),
#     ).images[0]

image = pipe(
        prompt=prompt, 
        num_inference_steps=28,
        guidance_scale=3.5,
        subject_image=concat,
        subject_scale=0.95,   
        generator=torch.manual_seed(seed),
    ).images[0]

image.save("flux_instantcharacter-2.png")
# pipe.unload_adapter()
# new_image = pipe(
#         prompt="do not look at the image at all, generate " + prompt, 
#         num_inference_steps=28,
#         guidance_scale=3.5,
#         subject_scale=0.9,
#         subject_image=Image.open("assets/100.jpg").convert('RGB'),
#         generator=torch.manual_seed(seed),
#     ).images[0]
# new_image.save("jghjg.png")