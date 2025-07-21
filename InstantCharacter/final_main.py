import uvicorn
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from PIL import Image
import io
import time
import gc
import torch

from final import (
    load_models_optimized, 
    text_to_image, 
    image_to_image, 
    create_avatar, 
    images_merge, 
    safety_checker,
    sketch_to_image
)

# Создаем FastAPI приложение с оптимизациями
app = FastAPI(
    title="Optimized Image Generation API",
    description="Сверхбыстрая генерация изображений за 5 секунд",
    version="2.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool для параллельной обработки
executor = ThreadPoolExecutor(max_workers=2)

# ---- Оптимизированные утилиты для конвертации ----
def pil_to_base64_fast(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Быстрая конвертация PIL в base64 с оптимизацией"""
    buffered = io.BytesIO()
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.save(buffered, format=format, quality=quality, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_pil_fast(data: str) -> Image.Image:
    """Быстрая конвертация base64 в PIL"""
    if data.startswith('data:image'):
        data = data.split(',', 1)[1]
    
    img_data = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_data))
    
    # Конвертируем в RGB для стабильности
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img

# --- Оптимизированные Pydantic модели ---
class PromptInput(BaseModel):
    prompt: str
    style_key: str

class AvatarInput(BaseModel):
    prompt: str

class MergeInput(BaseModel):
    images: list[str]
    prompt: str

class ImagePromptInput(BaseModel):
    image: str  # base64 string
    prompt: str
    style_key: str

class SafetyInput(BaseModel):
    image: str  # base64 string
class SketchInput(BaseModel):
    image: str
    prompt: str

# Декоратор для мониторинга производительности
def monitor_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"🚀 {func.__name__} выполнено за {execution_time:.2f} секунд")
        
        # Добавляем время выполнения в ответ
        if isinstance(result, dict) and "time" not in result:
            result["execution_time"] = execution_time
            
        return result
    return wrapper

# --- Startup с оптимизацией ---
@app.on_event("startup")
async def startup_event():
    """Инициализация с прогревом модели"""
    print("🚀 Запуск оптимизированного сервера...")
    
    # Очищаем память перед загрузкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Загружаем модель в отдельном потоке
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_models_optimized)
    
    print("✅ Сервер готов к работе!")

# --- Оптимизированные Endpoints ---

@app.post("/generate_from_text")
@monitor_performance
async def generate_from_text_optimized(request: PromptInput):
    """Сверхбыстрая генерация из текста"""
    try:
        # Запускаем генерацию в отдельном потоке
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            executor, 
            text_to_image, 
            request.prompt, 
            request.style_key
        )
        
        # Быстрая конвертация в base64
        image_b64 = pil_to_base64_fast(result_image)
        
        return {"image": image_b64, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в generate_from_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_avatar")
@monitor_performance
async def create_avatar_optimized(request: AvatarInput):
    """Сверхбыстрое создание аватара"""
    try:
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            executor, 
            create_avatar, 
            request.prompt
        )
        
        image_b64 = pil_to_base64_fast(result_image)
        return {"image": image_b64, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в create_avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/sketch_to_image")
@monitor_performance
async def sketch(request: SketchInput):
    """Сверхбыстрая генерация sketch 2 image"""
    try:
        # Конвертируем base64 в PIL
        input_image = base64_to_pil_fast(request.image)
        
        # Генерируем изображение
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            executor, 
            sketch_to_image, 
            input_image,
            request.prompt,
            
        )
        
        image_b64 = pil_to_base64_fast(result_image)
        return {"image": image_b64, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в generate_from_image_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/images_merge")
@monitor_performance
async def images_merge_optimized(request: MergeInput):
    """Сверхбыстрое слияние изображений"""
    try:
        # Конвертируем base64 в PIL параллельно
        def convert_image(img_b64):
            return base64_to_pil_fast(img_b64)
        
        loop = asyncio.get_event_loop()
        
        # Конвертируем все изображения параллельно
        pil_images = await asyncio.gather(*[
            loop.run_in_executor(executor, convert_image, img_b64)
            for img_b64 in request.images
        ])
        
        # Генерируем итоговое изображение
        result_image = await loop.run_in_executor(
            executor, 
            images_merge, 
            pil_images, 
            request.prompt
        )
        
        image_b64 = pil_to_base64_fast(result_image)
        return {"image": image_b64, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в images_merge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_from_image_text")
@monitor_performance
async def generate_from_image_text_optimized(request: ImagePromptInput):
    """Сверхбыстрая генерация из изображения и текста"""
    try:
        # Конвертируем base64 в PIL
        input_image = base64_to_pil_fast(request.image)
        
        # Генерируем изображение
        loop = asyncio.get_event_loop()
        result_image = await loop.run_in_executor(
            executor, 
            image_to_image, 
            request.prompt, 
            request.style_key, 
            input_image
        )
        
        image_b64 = pil_to_base64_fast(result_image)
        return {"image": image_b64, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в generate_from_image_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety")
async def safety_check_optimized(request: SafetyInput):
    """Быстрая проверка безопасности"""
    try:
        result = safety_checker(request.image)
        return {"data": result, "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в safety: {e}")
        return {"data": "good", "status": "error"}

# --- Дополнительные оптимизированные endpoints ---

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy", 
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }

@app.post("/batch_generate")
@monitor_performance
async def batch_generate(requests: list[PromptInput]):
    """Батчевая генерация для множественных запросов"""
    try:
        if len(requests) > 4:  # Ограничиваем размер батча
            raise HTTPException(status_code=400, detail="Максимум 4 запроса в батче")
        
        loop = asyncio.get_event_loop()
        
        # Обрабатываем все запросы параллельно
        tasks = [
            loop.run_in_executor(executor, text_to_image, req.prompt, req.style_key)
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Конвертируем в base64
        images_b64 = [pil_to_base64_fast(img) for img in results]
        
        return {"images": images_b64, "count": len(images_b64), "status": "success"}
        
    except Exception as e:
        print(f"❌ Ошибка в batch_generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Cleanup при завершении ---
@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при завершении"""
    print("🧹 Очищаем ресурсы...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    executor.shutdown(wait=True)

# --- Для запуска как скрипта ---
if __name__ == "__main__":
    uvicorn.run(
        "final_main:app", 
        host="0.0.0.0", 
        port=8003, 
        reload=False,  # Отключаем reload для производительности
        workers=1,     # Один worker для GPU модели
        loop="uvloop", # Быстрый event loop
        log_level="info"
    ) 