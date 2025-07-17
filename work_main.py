import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List
import base64
from PIL import Image
import io

from work_integrate import load_models, generate_image_from_request, generate_image_to_image

app = FastAPI()

# ---- Утилиты для конвертации изображений ----
def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(data: str) -> Image.Image:
    needed = data
    if needed.startswith('data:image'):
        needed = needed.split(',', 1)[1]
    print(data)
    return Image.open(io.BytesIO(base64.b64decode(needed)))

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
    load_models()
    # load_diff()
    # pass

# --- Endpoints ---
@app.post("/generate_from_text")
def generate_from_text(request: PromptInput):
    print(request)
    # Ваши generate_images_from_prompts ждёт список объектоnв с .prompt и .style_key атрибутами
    images = generate_image_from_request(request.prompt, request.style_key)
    images_base64 = pil_to_base64(images)
    return {"image": images_base64}

@app.post("/generate_from_image_text")
def generate_from_image_text(request: ImagePromptInput):
    # Преобразуем base64 картинки в PIL и готовим dict для вашей функции
    item = request
    image_prompt = {
        "image": base64_to_pil(item.image),
        "prompt": item.prompt,
        "style_key": item.style_key
    }
    print(image_prompt)
    images = generate_image_to_image(request.prompt, request.style_key, base64_to_pil(request.image))
    images_base64 = pil_to_base64(images)
    return {"image": images_base64}

# --- Для запуска как скрипта ---
if __name__ == "__main__":
    uvicorn.run("work_main:app", host="0.0.0.0", port=8003, reload=True)