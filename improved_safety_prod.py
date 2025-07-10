import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import sys
from huggingface_hub import login
login(token="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")
PROMPT_IMAGE_INTEGRITY = """
Task: Analyze an image to identify potential copyright concerns, depictions of public figures, or the presence of inappropriate, dangerous, or offensive content.

Output: Respond with only "yes" or "no"

Criteria for "yes":
- The image displays a trademarked logo or brand
- The image contains obscene, explicit, pornographic, or 18+ content (nudity, sexual acts, etc.)
- The image contains dangerous items or content (weapons, drugs, violence, suicide, self-harm, etc.)
- The image contains anything offensive or disrespectful to religious, political, or social groups

Criteria for "no":
- All other cases
- When you cannot identify the specific copyrighted work, named individual, or content listed above

Critical Requirements:
1. You must be able to name the exact copyrighted work, specific person depicted, or describe the prohibited content found
2. General references to demographics or characteristics are not sufficient
3. Base your decision solely on visual content, not interpretation
4. Provide only the one-word answer: "yes" or "no"
""".strip()

pipe = None  # глобальная переменная

def preload_model_safety():
    global pipe
    model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = (model, processor)

def generate_safety(image_list):
    """
    image_list: список PIL.Image
    Возвращает: список строк "good"/"bad"
    """
    global pipe
    if pipe is None:
        raise RuntimeError("Pipe is not initialized. Call init_pipe() first.")

    model, processor = pipe

    def get_text_after_substring(text, substring):
        index = text.find(substring)
        return text[index + len(substring):]

    results = []
    for image in image_list:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_IMAGE_INTEGRITY},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        output = model.generate(*inputs, max_new_tokens=200, do_sample=False)
        answer = processor.decode(output[0], skip_special_tokens=True).lower()
        answer = get_text_after_substring(answer, "assistant").strip()
        result = "bad" if "yes" in answer else "good"
        results.append(result)
    return results