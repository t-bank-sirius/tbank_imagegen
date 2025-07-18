import uvicorn
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List
import base64
from PIL import Image
import io
import time

from final import load_models, text_to_image, image_to_image, create_avatar, images_merge, safety_checker, sketch_to_image
from improved_safety_prod import preload_model_safety, generate_safety

app = FastAPI()

# ---- Утилиты для конвертации изображений ----
def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(data: str) -> Image.Image:
    needed = data
    print('needed')
    print(needed)
    if needed.startswith('data:image'):
        needed = needed.split(',', 1)[1]
    # print(data)
    return Image.open(io.BytesIO(base64.b64decode(needed)))

# --- Pydantic модели ---
class PromptInput(BaseModel):
    prompt: str
    style_key: str
class AvatarInput(BaseModel):
    prompt: str
class MergeInput(BaseModel):
    images: list[str]
    prompt: str
class SketchInput(BaseModel):
    prompt: str
    image: str



class ImagePromptInput(BaseModel):
    image: str  # base64 string
    prompt: str
    style_key: str
class SafetyInput(BaseModel):
    image: str  # base64 string

# --- Startup preload ---
@app.on_event("startup")
def load_model():
    load_models()
    # preload_model_safety()
    # load_diff()
    # pass

# --- Endpoints ---

@app.post("/generate_from_text")
def generate_from_text(request: PromptInput):
    start = time.time()
    
    # print(request)
    # print("processing")
    # Ваши generate_images_from_prompts ждёт список объектоnв с .prompt и .style_key атрибутами
    images = text_to_image(request.prompt, request.style_key)
    # if (generate_safety([images]) == "bad"):
    #     raise HTTPException(status_code=404, detail="")
    images_base64 = pil_to_base64(images)
    if safety_checker(images_base64) == "bad":
        return HTTPException(detail="Bad image")
    fin = time.time()
    return {"image": "data:image/jpeg;base64," +images_base64, "time": fin - start}

@app.post("/create_avatar")
def create_avatar_1(request: AvatarInput):
    start = time.time()
    st = request.prompt
    print('st')
    print(st)
    data = create_avatar(st)
    fin = time.time()
    return {"image": pil_to_base64(data), "time": fin - start}
@app.post("/sketch_to_image")
def sketch_to_image1(request: SketchInput):
    start = time.time()
    data = sketch_to_image(base64_to_pil(request.image), request.prompt)
    fin = time.time()
    return {"image": pil_to_base64(data), "time": fin - start}
@app.post("/images_merge")
def images_merge_1(request: MergeInput):
    # print(request)
    start = time.time()
    ims = []
    for el in request.images:
        ims.append(base64_to_pil(el))
        # print(ims)
    print('final_main')
    print(ims)
    print(request.prompt)
    data = images_merge(ims, request.prompt)
    fin = time.time()
    return {"image": pil_to_base64(data), "time": fin - start}
@app.post("/generate_from_image_text")
def generate_from_image_text(request: ImagePromptInput):
    start = time.time()
    # Преобразуем base64 картинки в PIL и готовим dict для вашей функции
    item = request
    # image_prompt = {
    #     "image": base64_to_pil(item.image),
    #     "prompt": item.prompt,
    #     "style_key": item.style_key
    # }
    # data = generate_safety([base64_to_pil(item.image)])
    # return {"data": data}
    # if (data == "bad"):
    #     print('yfyyyyv')
    #     raise HTTPException(status_code=404, detail="Bad image")
    # print(image_prompt)
    # print(generate_safety([base64_to_pil(item.image)]))
    # return {"data": generate_safety([base64_to_pil(item.image)])}
    images = image_to_image(request.prompt, request.style_key, base64_to_pil(request.image))
    images_base64 = pil_to_base64(images)
    fin = time.time()
    return {"image": "data:image/jpeg;base64,"+images_base64, "time": fin - start}

# --- Для запуска как скрипта ---
if __name__ == "__main__":
    uvicorn.run("final_main:app", host="0.0.0.0", port=8003, reload=True)