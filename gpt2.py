from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Model and Dataset IDs
model_name = "gpt2"
dataset_id = "HuggingFaceH4/CodeAlpaca_20K"

# Load model with optional 8-bit precision
use_8bit = False  
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    trust_remote_code=True,
    load_in_8bit=use_8bit,
)
model.config.use_cache = False

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Preprocess Dataset
def preprocess_dataset(dataset_id, tokenizer, split):
    dataset = load_dataset(dataset_id, split=split)

    def apply_prompt_template(sample):
        return {"prompt": sample["prompt"], "completion": sample["completion"]}

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_and_format(sample):
        max_length = 384
        prompt = tokenizer.encode(
            sample["prompt"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length // 2,
        )
        completion = tokenizer.encode(
            sample["completion"] + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length // 2,
        )
        input_ids = prompt + completion
        attention_mask = [1] * len(input_ids)

        # Pad sequences to max_length
        if len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        # Labels mask the prompt tokens with -100
        labels = [-100] * len(prompt) + completion[: len(input_ids) - len(prompt)]
        if len(labels) < max_length:
            labels += [-100] * (max_length - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = dataset.map(tokenize_and_format, remove_columns=list(dataset.features))
    return dataset

# Preprocess dataset
train_dataset = preprocess_dataset(dataset_id, tokenizer, "train")

# Configure LoRA
def configure_lora(model):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["attn.c_attn", "attn.c_proj"],
    )
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config

model, lora_config = configure_lora(model)

# Training Configuration
training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    logging_strategy="steps",
    save_strategy="steps",
    gradient_checkpointing=False,
)

# Initialize Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
    max_length=384,
    return_tensors="pt",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("gpt2")
tokenizer.save_pretrained("gpt2")
