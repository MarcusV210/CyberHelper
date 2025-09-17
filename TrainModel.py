from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore

model_name = "google/vaultgemma-1b"

# Load the pre-trained tokeniser and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Querying
text = "Tell me an unknown interesting biology fact about the brain."
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
out = pipe(text, max_new_tokens=32)
print(out[0]["generated_text"])