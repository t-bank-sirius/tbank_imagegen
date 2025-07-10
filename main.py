import io
import base64
from fastapi import FastAPI, UploadFile, File
from typing import List
from PIL import Image

from improved_safety_prod import preload_model_safety, generate_safety
from InstantCharacter.instant_prod import generate_image_to_image, preload_model_image_to_image
from prompt_to_image_prod import generate_images_from_prompts, preload_model_prompt_to_image

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

# --- Модели: загружаем при старте ---
@app.on_event("startup")
def preload_all_models():
    preload_model_safety()
    preload_model_image_to_image()
    preload_model_prompt_to_image()
# --------------------------

# --- Endpoint 1: Propmpt to Image ---
@app.post("/prompt-to-image/")
async def prompt_to_image_endpoint(prompts: List[str]):
    images = generate_images_from_prompts(prompts)
    # Возвращаем картинки как base64
    encoded_images = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        encoded_images.append(encoded)
    return {"images": encoded_images}

# --- Endpoint 2: Text & Image to Image ---
@app.post("/text-image-to-image/")
async def text_image_to_image_endpoint(
    prompts: List[str], 
    files: List[UploadFile] = File(...)
):
    images = []
    pil_images = []
    for uploaded in files:
        pil_images.append(Image.open(uploaded.file))
    # Передаём [{'image': ..., 'prompt': ...}, ...]
    items = [{"image": img, "prompt": p} for img, p in zip(pil_images, prompts)]
    generated = generate_image_to_image(items)
    # Кодируем в base64 для ответа
    for img in generated:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode()
        images.append(encoded)
    return {"images": images}

# --- Endpoint 3: Image Safety ---
@app.post("/check-image-safety/")
async def image_safety_endpoint(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = generate_safety(image)  # возвращает good/bad
    return {"status": result}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)