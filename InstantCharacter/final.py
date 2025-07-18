import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from style_prompt import get_prompt
import requests
pipe = None
def load_models():
    global pipe
    ip_adapter_path = '/workspaces/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
    base_model = "/workspaces/tbank_imagegen/models/flux-8bit"
    image_encoder_path = '/workspaces/tbank_imagegen/models/google'
    image_encoder_2_path = '/workspaces/tbank_imagegen/models/facebook'
    # dtype = torch.bfloat16

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
def concatenate_images(images, direction="horizontal"):
    """
    Concatenate multiple PIL images either horizontally or vertically.
    
    Args:
        images: List of PIL Images
        direction: "horizontal" or "vertical"
    
    Returns:
        PIL Image: Concatenated image
    """
    if not images:
        return None
    
    # Filter out None images
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    # Convert all images to RGB
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  # vertical
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in valid_images:
            # Center image horizontally if widths differ
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated
def safety_checker(image: str):
    r = requests.post("http://localhost:8002/analyze", {"image_base64": image, "comment": ""})
    if r.content.result == "This image contains potentially unsafe or inappropriate content and cannot be analyzed due to safety restrictions.":
        return "bad"
    else:
        return "good"
def sketch_to_image(image: Image, prompt: str):
    new_image = pipe(
        prompt=prompt + " Create a colorful, friendly, high-quality illustration based on this child's sketch.", 
        num_inference_steps=16,
        guidance_scale=3.5,
        subject_scale=0.9,
        height=512,
        width=512,
        subject_image=image
    ).images[0]
    return new_image
def create_avatar(prompt: str):
    print("prompt")
    print(prompt)
    new_image = pipe(
        prompt=prompt + " A cute 3D character portrait in Pixar Disney style, soft lighting, big expressive eyes, friendly smile, pastel colors, upper body shot, studio background, highly detailed, smooth skin, vibrant and appealing look — perfect for an avatar", 
        num_inference_steps=8,
        guidance_scale=3.5,
        subject_scale=0.9,
        height=256,
        width=256,
        subject_image=Image.open("/workspaces/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
    ).images[0]
    return new_image
def images_merge(images: list[str], prompt: str):
    print('jhbhhhv')
    print(images)
    valid_images = images
    concatenated_image = concatenate_images(valid_images, "horizontal")
    original_width, original_height = concatenated_image.size
    new_width = None
    new_height = None
    if original_width >= original_height:
        new_width = 512
        new_height = int(original_height * (new_width / original_width))
        new_height = round(new_height / 64) * 64
    else:
        new_height = 512
        new_width = int(original_width * (new_height / original_height))
        new_width = round(new_width / 64) * 64
    
    concatenated_image_resized = concatenated_image.resize((new_width, new_height), Image.LANCZOS)

    final_prompt = f'From the provided reference images, create a unified, cohesive image that satisfies this prompt: {prompt}. Maintain the identity and characteristics of each subject while adjusting their proportions, scale, and positioning to create a harmonious, naturally balanced composition. Blend and integrate all elements seamlessly with consistent lighting, perspective, and style.the final result should look like a single naturally captured scene where all subjects are properly sized and positioned relative to each other, not assembled from multiple sources.'
    
    image = pipe(
        subject_image=concatenated_image, 
        prompt=final_prompt,
        guidance_scale=0.9,
        width=concatenated_image_resized.size[0],
        num_inference_steps=31,
        height=concatenated_image_resized.size[1],
    ).images[0]
    return image
def text_to_image(prompt: str, style_key: str):
    # pipe.unload_adapter()
    new_image = pipe(
        prompt="do not look at the image at all, just generate this in high resolution: " + get_prompt(prompt, style_key), 
        num_inference_steps=9,
        guidance_scale=3.5,
        subject_scale=0.9,
        height=512,
        width=512,
        subject_image=Image.open("/workspaces/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
    ).images[0]
    return new_image

def image_to_image(prompt: str, style_key: str, image: Image):
    new_image = pipe(
        prompt=get_prompt(prompt, style_key), 
        num_inference_steps=16,
        guidance_scale=3.5,
        subject_scale=0.9,
        subject_image=image,
        height=512,
        width=512,
    ).images[0]
    return new_image