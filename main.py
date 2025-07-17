import io
import base64
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from typing import List
from PIL import Image
from pydantic import BaseModel
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

class PromptModel(BaseModel):
    prompt: str
    style_key: str

class PromptImageModel(BaseModel):
    prompt: str
    image: str  # base64-строка
    style_key: str

@app.post("/text-to-image/")
async def text_to_image_endpoint(body: PromptModel):
    result_images = generate_images_from_prompts([{"prompt": body.prompt, "style_key": body.style_key}])
    safety_status = generate_safety(result_images)
    if safety_status != 'good':
        raise HTTPException(status_code=400, detail="Generated image is unsafe")
    buf = io.BytesIO()
    result_images[0].save(buf, format='PNG')
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return {"image": encoded_image}

@app.post("/image-text-to-image/")
async def image_text_to_image_endpoint(body: PromptImageModel):
    image_bytes = base64.b64decode(body.image)
    pil_image = Image.open(io.BytesIO(image_bytes))
    input_safety_status = generate_safety([pil_image])
    if input_safety_status != 'good':
        raise HTTPException(status_code=400, detail="Input image is unsafe")
    items = [{"image": pil_image, "prompt": body.prompt}]
    result_images = generate_image_to_image(items)
    output_safety_status = generate_safety(result_images)
    if output_safety_status != 'good':
        raise HTTPException(status_code=400, detail="Generated image is unsafe")
    buf = io.BytesIO()
    result_images[0].save(buf, format='PNG')
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    return encoded_image

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)