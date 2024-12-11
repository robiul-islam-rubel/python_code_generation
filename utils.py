from datasets import load_dataset, Dataset,DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments,AutoModelWithLMHead,BitsAndBytesConfig,AutoModelForSeq2SeqLM,GPT2LMHeadModel, GPT2Tokenizer,DataCollatorForSeq2Seq
from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
        PeftModel
    )
import torch
import pandas as pd
import re
from tqdm import tqdm
import sacrebleu  
from rouge_score import rouge_scorer
import datasets
from bert_score import score as bert_score
from transformers import default_data_collator, Trainer
import evaluate
bleu_metric = evaluate.load("sacrebleu")


def get_dataset(dataset):
    dataset = load_dataset(f"{dataset}", "python", trust_remote_code=True)
    print(f"Dataset size: {len(dataset)}")
    return dataset

def remove_columns(dataset):
    # If the input is a DatasetDict, process each split
    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            # Convert to Pandas and process each split
            df = dataset[split].to_pandas()

            # Rename and keep required columns
            if "func_documentation_string" in df.columns:
                df.rename(columns={"func_documentation_string": "docstring"}, inplace=True)
            else:
                print(f"Column 'func_documentation_string' not found in the {split} split")

            if "func_code_string" in df.columns:
                df.rename(columns={"func_code_string": "code"}, inplace=True)
            else:
                raise ValueError(f"Required column 'func_code_string' not found in the {split} split")

            # Keep only necessary columns
            df = df[["code", "docstring"]]

            # Convert back to Dataset
            dataset[split] = Dataset.from_pandas(df)
    else:
        # Single dataset processing
        df = dataset.to_pandas()

        # Rename and keep required columns
        if "func_documentation_string" in df.columns:
            df.rename(columns={"func_documentation_string": "docstring"}, inplace=True)
        else:
            print("Column 'func_documentation_string' not found in the dataset")

        if "func_code_string" in df.columns:
            df.rename(columns={"func_code_string": "code"}, inplace=True)
        else:
            raise ValueError("Required column 'func_code_string' not found in the dataset")

        # Keep only necessary columns
        df = df[["code", "docstring"]]

        # Convert back to Dataset
        dataset = Dataset.from_pandas(df)

    return dataset

def remove_comment(dataset):
    def clean_split(split_data):
        # Convert to Pandas DataFrame
        data = split_data.to_pandas()

        # Define regex patterns
        single_line_comment_pattern = r"#.*"
        multi_line_comment_pattern = r'(?s)(?:(?:r|R)?"""(?:.*?)"""|\'\'\'(?:.*?)\'\'\')'

        # Function to remove comments
        def remove_comments(code):
            code_no_multiline = re.sub(multi_line_comment_pattern, '', code, flags=re.DOTALL)
            code_no_comments = re.sub(single_line_comment_pattern, '', code_no_multiline)
            return code_no_comments.strip()

        # Apply to `func_code_string` column
        if "code" in data.columns:
            data["code"] = data["code"].apply(remove_comments)
        else:
            raise ValueError("Column 'func_code_string' not found in the dataset")

        # Convert back to Dataset
        return Dataset.from_pandas(data)

    # Handle DatasetDict or single Dataset
    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            dataset[split] = clean_split(dataset[split])
    else:
        dataset = clean_split(dataset)

    print("Comments Removed")
    return dataset

def filter_dataset(dataset):
    dataset = remove_columns(dataset)
    dataset = remove_comment(dataset)
    return dataset

def set_device(num):
    # Check if GPU is available
    use_GPU = torch.cuda.is_available()
    
    if use_GPU:
        # Validate the requested GPU number
        # if num <= torch.cuda.device_count():
        #     print(num,torch.cuda.device_count())
        #     raise ValueError(f"Invalid GPU number: {num}. Only {torch.cuda.device_count()} GPUs available.")

        device = f"cuda:{num}"
        torch.cuda.set_device(device)  

        print("Device:", device)
        print("GPU Name:", torch.cuda.get_device_name(num))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(num) / 1024 ** 3, 1), "GB")
        print("Reserved:", round(torch.cuda.memory_reserved(num) / 1024 ** 3, 1), "GB")
    else:
        device = "cpu"
        print("Using CPU")

    return device
