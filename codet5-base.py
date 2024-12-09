from utils import *

# Model and Dataset IDs
model_name = "Salesforce/codet5-base"
dataset_id = "HuggingFaceH4/CodeAlpaca_20K"

# Load model
model = AutoModelWithLMHead.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
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
def get_preprocessed_cmg_history(dataset_id, tokenizer, split):
    dataset = datasets.load_dataset(dataset_id, split=split)

    # Apply prompt template
    def apply_prompt_template(sample):
        return {
            "prompt": sample["prompt"],
            "message": sample["completion"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    # Tokenize and add labels
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["prompt"], 
            add_special_tokens=False, 
            max_length=200, 
            truncation=True
        )
        message = tokenizer.encode(
            sample["message"] + tokenizer.eos_token, 
            max_length=400, 
            truncation=True, 
            add_special_tokens=False
        )
        max_length = 601 - len(prompt) - len(message)

        # Handle cases where max_length is negative
        if max_length < 0:
            pad = []
        else:
            pad = tokenizer.encode(
                tokenizer.eos_token, 
                add_special_tokens=False, 
                max_length=max_length, 
                padding='max_length', 
                truncation=True
            )

        sample = {
            "input_ids": prompt + message + pad,
            "attention_mask": [1] * (len(prompt) + len(message) + len(pad)),
            "labels": [-100] * len(prompt) + message + [-100] * len(pad),
        }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

# Load preprocessed training dataset
train_dataset = get_preprocessed_cmg_history(dataset_id, tokenizer, 'train')

# Create LoRA Configuration
def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=4,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=[
            "q",
            "v",
        ],
    )

    # Prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# Configure LoRA model
model, lora_config = create_peft_config(model)

# Training Configuration
# batch_size = 128
per_device_train_batch_size = 4
# gradient_accumulation_steps = batch_size // per_device_train_batch_size

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=0.5,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=2, 
    optim="adamw_torch",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    logging_strategy="steps",
    save_strategy="no",
    gradient_checkpointing=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("codet5-base")
tokenizer.save_pretrained("codet5-base")
