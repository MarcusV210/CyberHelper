from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore
import os
from dotenv import load_dotenv #type: ignore

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# print(HF_TOKEN)

model_name = "google/gemma-2b"

# Load the pre-trained tokeniser and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
    )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    torch_dtype="auto", 
    device_map="auto", 
    trust_remote_code=True
    )

# Querying
text = "Tell me an unknown interesting biology fact about the brain."
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
out = pipe(text, max_new_tokens=32)
print(out[0]["generated_text"])