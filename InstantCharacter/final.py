import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from style_prompt import get_prompt
pipe = None
def load_models():
    global pipe
    ip_adapter_path = '/workspaces/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
    base_model = pipe = "/workspaces/tbank_imagegen/models/flux-instant-3"
    image_encoder_path = '/workspaces/tbank_imagegen/models/google'
    image_encoder_2_path = '/workspaces/tbank_imagegen/models/facebook'
    dtype = torch.bfloat16

    pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

    pipe.to("cpu")
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
    pipe.enable_sequential_cpu_offload()

    pipe.init_adapter(
        image_encoder_path=image_encoder_path, 
        image_encoder_2_path=image_encoder_2_path, 
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
        device=torch.device('cuda')
    )
def text_to_image(prompt: str, style_key: str):
    # pipe.unload_adapter()
    new_image = pipe(
        prompt="do not look at the image at all, just generate this in high resolution: " + get_prompt(prompt, style_key), 
        num_inference_steps=8,
        guidance_scale=3.5,
        subject_scale=0.9,
        subject_image=Image.open("/workspaces/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
    ).images[0]
    return new_image

def image_to_image(prompt: str, style_key: str, image: Image):
    new_image = pipe(
        prompt=get_prompt(prompt, style_key), 
        num_inference_steps=8,
        guidance_scale=3.5,
        subject_scale=0.9,
        subject_image=image
    ).images[0]
    return new_image