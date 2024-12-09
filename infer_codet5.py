from utils import *

device = set_device(0)

# Load the tokenizer and model
model_name = "codet5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelWithLMHead.from_pretrained(model_name, trust_remote_code=True)

model.to(device)

# Ensure padding token is correctly set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Test function with Exact Match and BLEU, BERTScore, ROUGE
def test_model_with_metrics(model, tokenizer, test_data):
    results = []
    exact_match_count = 0
    hypotheses = []
    references = []

    for example in tqdm(test_data):
        instruction = example["prompt"].strip()
        expected_output = example["completion"].strip()

        # Skip examples with empty prompt or completion
        if not instruction or not expected_output:
            continue

        print(instruction, "******************", expected_output)

        # Tokenize and generate output
        inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                num_return_sequences=1,
                num_beams=1,
                do_sample=False,
                attention_mask=attention_mask,
            )
        generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Handle empty generation
        if not generated_output:
            generated_output = "EMPTY"

        # Check for exact match
        is_exact_match = generated_output == expected_output
        if is_exact_match:
            exact_match_count += 1

        # Collect results
        results.append({
            "instruction": instruction,
            "expected_output": expected_output,
            "generated_output": generated_output,
            "exact_match": is_exact_match
        })
        hypotheses.append(generated_output)
        references.append(expected_output)

        print(f"Instruction: {instruction}")
        print(f"Expected Output: {expected_output}")
        print(f"Generated Output: {generated_output}")
        print(f"Exact Match: {is_exact_match}")
        print("-" * 50)

    # Compute exact match percentage
    exact_match_percentage = (exact_match_count / len(test_data)) * 100 if test_data else 0
    print(f"Exact Match Percentage: {exact_match_percentage:.2f}%")

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU Score: {bleu.score:.2f}")

    # Compute ROUGE score
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    for ref, hyp in zip(references, hypotheses):
        scores = rouge_scorer_instance.score(ref, hyp)
        rouge_scores["rouge1"] += scores["rouge1"].fmeasure
        rouge_scores["rouge2"] += scores["rouge2"].fmeasure
        rouge_scores["rougeL"] += scores["rougeL"].fmeasure

    num_examples = len(references)
    for key in rouge_scores:
        rouge_scores[key] /= num_examples
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    # Compute BERTScore
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=True)
    print(f"BERTScore Precision: {P.mean().item():.4f}")
    print(f"BERTScore Recall: {R.mean().item():.4f}")
    print(f"BERTScore F1: {F1.mean().item():.4f}")

    return results, exact_match_percentage, bleu.score, rouge_scores, F1.mean().item()




if __name__=="__main__":
 
 dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", trust_remote_code=True)
 print(dataset)
 test_data = dataset['test']
 
 print(test_data[0])
 # Evaluate the model
results, exact_match_percentage, bleu_score, rouge_scores, bert_f1 = test_model_with_metrics(model, tokenizer, test_data)

# Save results to CSV
pd.DataFrame(results).to_csv("test_results_with_metrics.csv", index=False)

