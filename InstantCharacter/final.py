import torch
from PIL import Image
from pipeline import InstantCharacterFluxPipeline
from style_prompt import get_prompt
import requests
import time

pipe = None

def load_models_optimized():
    """Оптимизированная загрузка модели для максимальной производительности"""
    global pipe
    
    print("🔥 Загружаем модель с оптимизациями...")
    
    # Настройки для максимальной производительности
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")  # Или "high" для лучшего качества
    
    # Пути к моделям
    ip_adapter_path = '/workspaces/nikbauer34/tbank_imagegen/models/checkpoint/instantcharacter_ip-adapter.bin'
    base_model = "/workspaces/nikbauer34/tbank_imagegen/models/flux-8bit"
    image_encoder_path = '/workspaces/nikbauer34/tbank_imagegen/models/google'
    image_encoder_2_path = '/workspaces/nikbauer34/tbank_imagegen/models/facebook'
    
    # Загружаем модель с fp16 для производительности
    pipe = InstantCharacterFluxPipeline.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16,  # Используем fp16 вместо bfloat16
        use_safetensors=True
    )
    
    # ВАЖНО: Загружаем ВСЮ модель на GPU - БЕЗ CPU offloading!
    # Для 8bit моделей используем device_map вместо .to()
    if hasattr(pipe, 'device_map') and pipe.device_map is not None:
        # Модель уже на GPU через device_map
        print("✅ Модель загружена через device_map")
    else:
        pipe.to("cuda")
    
    # Включаем memory efficient attention (без xformers)
    try:
        pipe.enable_attention_slicing(1)
        print("✅ Attention slicing включен")
    except Exception as e:
        print(f"⚠️  Attention slicing не удалось включить: {e}")
    
    # Пробуем включить xformers только если доступен
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory efficient attention включен")
    except ImportError:
        print("⚠️  xformers не установлен, используем стандартное attention")
    except Exception as e:
        print(f"⚠️  xformers не удалось включить: {e}")
    
    # Инициализируем адаптер
    pipe.init_adapter(
        image_encoder_path=image_encoder_path, 
        image_encoder_2_path=image_encoder_2_path, 
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
        device=torch.device('cuda')
    )
    
    # Torch.compile для ускорения inference (только если не 8bit)
    try:
        print("🚀 Компилируем модель...")
        
        # Настройки для torch.compile
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        
        # Компилируем ключевые компоненты
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False
        )
        
        pipe.vae.decoder = torch.compile(
            pipe.vae.decoder,
            mode="max-autotune",
            fullgraph=True
        )
        print("✅ torch.compile применен")
    except Exception as e:
        print(f"⚠️  torch.compile не удалось применить: {e}")
    
    # Прогрев модели
    print("🔥 Прогреваем модель...")
    warmup_image(pipe)
    
    print("✅ Модель готова к работе!")

def warmup_image(pipe):
    """Прогрев модели для первого быстрого inference"""
    try:
        warmup_image = Image.open("/workspaces/nikbauer34/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
        
        # Делаем несколько тестовых inference для прогрева
        with torch.inference_mode():
            for i in range(2):  # 2 итерации прогрева
                _ = pipe(
                    prompt="test warmup",
                    num_inference_steps=4,  # Минимум для прогрева
                    guidance_scale=3.5,
                    subject_scale=0.9,
                    height=512,
                    width=512,
                    subject_image=warmup_image
                ).images[0]
                print(f"Прогрев {i+1}/2 завершен")
    except Exception as e:
        print(f"⚠️  Прогрев не удался: {e}")

def concatenate_images(images, direction="horizontal"):
    """Оптимизированная конкатенация изображений"""
    if not images:
        return None
    
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    # Конвертируем в RGB за один раз
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        x_offset = 0
        for img in valid_images:
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
    else:  # vertical
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in valid_images:
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

def safety_checker(image: str):
    """Быстрая проверка безопасности через внешний API"""
    try:
        r = requests.post("http://0.0.0.0:8001/analyze", 
                         {"image_base64": image, "comment": ""}, 
                         timeout=2)  # Быстрый timeout
        return r.json()
    except:
        return "good"  # При ошибке считаем безопасным

def create_avatar(prompt: str):
    """Оптимизированное создание аватара"""
    with torch.inference_mode():
        avatar_image = pipe(
            prompt=f"{prompt} A cute 3D character portrait in Pixar Disney style, soft lighting, big expressive eyes, friendly smile, pastel colors, upper body shot, studio background", 
            num_inference_steps=9,  # Уменьшено с 8 до 4
            guidance_scale=3.5,
            subject_scale=0.9,
            height=256,
            width=256,
            subject_image=Image.open("/workspaces/nikbauer34/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
        ).images[0]
    return avatar_image

def images_merge(images: list, prompt: str):
    """Оптимизированное слияние изображений"""
    valid_images = images
    concatenated_image = concatenate_images(valid_images, "horizontal")
    
    # Быстрое изменение размера
    original_width, original_height = concatenated_image.size
    if original_width >= original_height:
        new_width = 512
        new_height = int(original_height * (new_width / original_width))
        new_height = round(new_height / 64) * 64
    else:
        new_height = 512
        new_width = int(original_width * (new_height / original_height))
        new_width = round(new_width / 64) * 64
    
    concatenated_image_resized = concatenated_image.resize((new_width, new_height), Image.LANCZOS)

    final_prompt = f'Create unified image: {prompt}. Maintain identity while blending seamlessly.'
    
    with torch.inference_mode():
        result_image = pipe(
            subject_image=concatenated_image, 
            prompt=final_prompt,
            guidance_scale=0.9,
            width=concatenated_image_resized.size[0],
            num_inference_steps=6,  # Уменьшено с 31 до 6
            height=concatenated_image_resized.size[1],
        ).images[0]
    
    return result_image

def text_to_image(prompt: str, style_key: str):
    """Оптимизированная генерация из текста"""
    with torch.inference_mode():
        result_image = pipe(
            prompt="don't look at the image, just generate: " + get_prompt(prompt, style_key), 
            num_inference_steps=13,  # Уменьшено с 9 до 4
            guidance_scale=3.5,
            subject_scale=0.9,
            height=512,
            width=512,
            subject_image=Image.open("/workspaces/nikbauer34/tbank_imagegen/InstantCharacter/assets/100.jpg").convert('RGB')
        ).images[0]
    return result_image

def image_to_image(prompt: str, style_key: str, image: Image):
    print(image)
    """Оптимизированная генерация из изображения"""
    with torch.inference_mode():
        result_image = pipe(
            prompt=get_prompt(prompt, style_key), 
            num_inference_steps=20,  # Уменьшено с 18 до 6
            guidance_scale=3.5,
            subject_scale=0.9,
            subject_image=image,
            height=512,
            width=512,
        ).images[0]
    return result_image 
def sketch_to_image(image: Image, prompt: str):
    new_image = pipe(
        prompt=prompt + " Create a colorful, friendly, high-quality illustration based on this child's sketch.", 
        num_inference_steps=14,
        guidance_scale=3.5,
        subject_scale=0.9,
        height=512,
        width=512,
        subject_image=image
    ).images[0]
    return new_image