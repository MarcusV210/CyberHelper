from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer #type: ignore
from dotenv import load_dotenv #type: ignore
from datasets import load_dataset  #type: ignore
from peft import LoraConfig, get_peft_model  #type: ignore
import torch  #type: ignore
import os

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
    dtype="auto", 
    device_map="auto", 
    trust_remote_code=True
    )
print("1. Model and tokenizer loaded.")


dataset = load_dataset('csv', data_files={'train':'data/CyberSecurity_data.csv'})
def format_conv(a):
    text = f"### Prompt:\n{a['INSTRUCTION']}\n\n### Response:\n{a['RESPONSE']}"
    return {"text": text}

dataset = dataset.map(format_conv)
print("2. Dataset loaded and formatting finished.")


def tokenize(a):
    return tokenizer(
        a['text'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

tokenized_dataset = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)
print("3. Dataset batched and tokenzied.")


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # works for most transformer LMs
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("4. LoRA applied.")


training_args = TrainingArguments(
    output_dir="./gemma-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    processing_class=tokenizer,
)
print("5. Model training started.")

trainer.train()

trainer.save_model("./gemma-finetuned")