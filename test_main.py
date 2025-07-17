import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List
import base64
from PIL import Image
import io

from integrate import preload_model, generate_images_from_prompts, generate_images_from_image_and_prompt

app = FastAPI()

# ---- Утилиты для конвертации изображений ----
def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))

# --- Pydantic модели ---
class PromptInput(BaseModel):
    prompt: str
    style_key: str


class ImagePromptInput(BaseModel):
    image: str  # base64 string
    prompt: str
    style_key: str


# --- Startup preload ---
@app.on_event("startup")
def load_model():
    preload_model()
    # pass

# --- Endpoints ---
@app.post("/generate_from_text")
def generate_from_text(request: PromptInput):
    # Ваши generate_images_from_prompts ждёт список объектоnв с .prompt и .style_key атрибутами
    images = generate_images_from_prompts([request])
    images_base64 = [pil_to_base64(img) for img in images]
    return {"images": images_base64}

@app.post("/generate_from_image_text")
def generate_from_image_text(request: ImagePromptInput):
    # Преобразуем base64 картинки в PIL и готовим dict для вашей функции
    item = request
    image_prompt = {
        "image": base64_to_pil(item.image),
        "prompt": item.prompt,
        "style_key": item.style_key
    }
    images = generate_images_from_image_and_prompt([image_prompt])
    images_base64 = [pil_to_base64(img) for img in images]
    return {"images": images_base64}

# --- Для запуска как скрипта ---
if __name__ == "__main__":
    uvicorn.run("test_main:app", host="0.0.0.0", port=8002, reload=True)