from utils import *
from datasets import DatasetDict

model_name = "codellama/CodeLlama-7b-Python-hf"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_8bit=True
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    input_texts = examples["docstring"]
    target_texts = examples["code"]

    model_inputs = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        target_texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )["input_ids"]

    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_list]
        for label_list in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs


def create_peft_config(model):
    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.01,
        target_modules = ["q_proj", "v_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, peft_config


model, lora_config = create_peft_config(model)


def train_model(dataset):
    print("Finetuning Model ...")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Sample a subset of the dataset (e.g., 50,000 samples)
    # train_dataset = train_dataset.shuffle(seed=42).select(range(20000))
    # test_dataset = test_dataset.shuffle(seed=42).select(range(500))


    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['code', 'docstring'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['code', 'docstring'])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=True,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("codellama")
    tokenizer.save_pretrained("codellama")


if __name__ == "__main__":
    dataset = get_dataset("code-search-net/code_search_net")
    dataset = remove_columns(dataset)
    train_model(dataset)
