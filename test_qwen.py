import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_PATH = "models/Qwen3.5-2B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    local_files_only=True,
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Reply with exactly: Qwen is running."}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

outputs = model.generate(**inputs, max_new_tokens=80)
generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
raw_answer = processor.decode(generated_tokens, skip_special_tokens=False)
answer = processor.decode(generated_tokens, skip_special_tokens=True)

print("Raw output:", repr(raw_answer))
print("Answer:", repr(answer))
