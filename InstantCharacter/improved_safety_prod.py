import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import sys
from huggingface_hub import login
# login(token="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")
# PROMPT_IMAGE_INTEGRITY = """
# Task: Analyze an image to identify the presence of inappropriate, dangerous, or offensive content.

# Output: Respond with only "okay" or "bad" 
# Criteria for "bad":
# - The image contains obscene, explicit, pornographic, or 18+ content (nudity, sexual acts, etc.)
# - The image contains dangerous items or content (weapons, drugs, violence, suicide, self-harm, etc.). If there is something from this list, it automatically shoud be counted as a dangerous content!

# Criteria for "okay":
# - All other cases
# - If there is just a character or a human without anything explicit
# - When you cannot identify content listed above
# Examples:
# 1. There is a pistol or a knife or a rifle - "bad"
# 2. There is some white medicine - "bad"
# 3. Boy / girl /fictional character image - "okay"
# Critical Requirements:
# 2. General references to demographics or characteristics are not sufficient
# 3. Base your decision solely on visual content, not interpretation
# 5. If there is just a character or a human without anything explicit, then it is "okay"
# 4. Respond with only "okay" or "bad"
# """.strip()
PROMPT_IMAGE_INTEGRITY = """
Task: Analyze this image and write only "okay" or "bad"
Answer "bad" if:
- The image contains nudity, sexual acts, etc.
- The image contains weapons, firearms, riffles, drugs, violence, suicide, self-harm, etc.
Answer "okay" all other times
""".strip()

pipe = None  # глобальная переменная

def preload_model_safety():
    global pipe
    model_id = "/workspaces/tbank_imagegen/models/qwen"
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
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = processor.decode(output[0], skip_special_tokens=True).lower()
        answer = get_text_after_substring(answer, "assistant").strip()
        # return answer
        print("answer")
        print(answer)
        result = "good" if "okay" in answer else "bad"
        results.append(result)
    if results[0] == "good":
        return "good"
    else:
        return "bad"
    # return results