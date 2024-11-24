import json
import torch
from modeling_llama import LlamaForCausalLM
from tokenization_llama_fast import LlamaTokenizerFast
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os
from collections import Counter
from huggingface_hub import snapshot_download
import csv
import logging


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Large Language Model Multiple Choice QA with Majority Voting')

    # Data arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file (train/valid/test)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output predictions JSONL file')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or model name in HuggingFace hub')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu). If None, automatically detected')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for generation')

    # Voting arguments
    parser.add_argument('--num_forward_passes', type=int, default=10, help='Number of forward passes for majority voting')
    parser.add_argument('--min_probability_threshold', type=float, default=0.1, help='Minimum probability threshold for a prediction to be considered')

    # Other arguments
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_model_downloaded(model_path):
    """
    Ensure the model is available locally. If not, download it.
    Returns the path to the model.
    """
    if os.path.exists(model_path):
        return model_path

    try:
        print(f"Downloading model unsloth/Llama-3.2-1B-Instruct to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        snapshot_download(
            repo_id="unsloth/Llama-3.2-1B-Instruct",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        return model_path

    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def load_data(file_path):
    """
    Load data from JSONL file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_prompt(question, choices, tokenizer):
    """
    Create a formatted prompt for the model.
    """
    prompt = f"Question: {question}\n\nChoices:\n"

    for label, text in zip(choices['label'], choices['text']):
        prompt += f"{label}) {text}\n"

    prompt += "\nProvide a very brief rationale then state 'Answer: X' where X is one of the option letters (A, B, C, D, or E)."

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt, choices['label']


def get_next_token_probabilities(model, tokenizer, prompt, options, device, temperature, max_new_tokens=64):
    """
    Generate rationale and get probabilities for the final answer.
    """
    # First generate the rationale and "Answer: " prefix
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate until we get to "Answer: "
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.6,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Now create a new prompt with the generated rationale
        final_prompt = generated_text + "\nAnswer:"

        # Get the final answer probabilities
        inputs = tokenizer(final_prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Get option tokens
        option_tokens = {}
        for opt in options:
            tokens = tokenizer.encode(f" {opt}")
            option_tokens[opt] = tokens[1]

        # Extract logits only for the option tokens
        option_logits = logits[0, [token_id for token_id in option_tokens.values()]]

        # Apply softmax to get probabilities
        option_probs = F.softmax(option_logits, dim=-1)

        # Convert to dictionary
        option_probs = {
            opt: option_probs[i].item()
            for i, opt in enumerate(option_tokens.keys())
        }

    return option_probs, generated_text


def majority_vote(all_predictions, min_probability_threshold):
    """
    Perform majority voting across multiple predictions.
    """
    filtered_predictions = []
    rationales = []

    for preds, rationale in all_predictions:
        # Only consider predictions above the threshold
        valid_preds = {k: v for k, v in preds.items() if v >= min_probability_threshold}
        if valid_preds:
            max_pred = max(valid_preds.items(), key=lambda x: x[1])[0]
            filtered_predictions.append(max_pred)
            rationales.append(rationale)

    if not filtered_predictions:
        return "A", 0.0, ""  # Default fallback with confidence 0

    # Count occurrences of each prediction
    vote_counts = Counter(filtered_predictions)
    total_votes = len(filtered_predictions)

    # Get the majority prediction and its confidence
    majority_pred = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_pred] / total_votes

    # Select the rationale from the most confident prediction
    best_rationale = rationales[filtered_predictions.index(majority_pred)]

    return majority_pred, confidence, best_rationale

def save_results(results, output_file):
    """
    Save results to JSONL file.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


def main():

    # Parse arguments and set seed
    args = parse_arguments()
    set_seed(args.seed)
    print(f"Using device: {args.device}")

    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_data(args.input_file)

    # Use only 25% of the data
    data = data[:len(data) // 4]

    # Ensure model is downloaded and get the correct path
    try:
        model_path = ensure_model_downloaded(args.model_path)
        print(f"Using model from: {model_path}")
    except Exception as e:
        print(f"Error ensuring model availability: {e}")
        return

    # Initialize model and tokenizer
    print(f"Loading model...")
    try:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto"
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path, truncation_side="left")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup logging
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    temperature = args.temperature  # Extracted from model.generate
    log_filename = f"log-{temperature}-{args.num_forward_passes}.csv"
    log_path = os.path.join(logs_dir, log_filename)
    log_exists = os.path.isfile(log_path)
    if not log_exists:
        log_file = open(log_path, 'a', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['len', 'accuracy'])
        log_file.close()
    

    # Make predictions
    results = []
    correct_predictions = 0
    total_questions = len(data)

    for item in tqdm(data, desc="Processing questions"):
        prompt, options = create_prompt(item['question'], item['choices'], tokenizer)

        try:
            # Collect predictions and rationales from multiple forward passes
            all_predictions = []
            for _ in range(args.num_forward_passes):
                probs, rationale = get_next_token_probabilities(
                    model, tokenizer, prompt, options, args.device, args.temperature)
                all_predictions.append((probs, rationale))

            # Perform majority voting
            prediction, confidence, best_rationale = majority_vote(
                all_predictions,
                args.min_probability_threshold
            )

        except Exception as e:
            print(f"Error with prediction: {e}")
            prediction = "A"
            confidence = 0.0
            best_rationale = ""
            all_predictions = []

        # Store result
        result = {
            'id': item['id'],
            'question': item['question'],
            'predicted': prediction,
            'confidence': confidence,
            'rationale': best_rationale,
            'individual_predictions': [p[0] for p in all_predictions]  # Store only probabilities
        }

        # Add actual answer and correctness if available
        if 'answerKey' in item:
            result['actual'] = item['answerKey']
            result['correct'] = prediction == item['answerKey']
            if result['correct']:
                correct_predictions += 1

        results.append(result)

        # Print intermediate results for long runs
        if (len(results) % 20) == 0:
            print(f"\nProcessed {len(results)}/{total_questions} questions")
            if 'answerKey' in data[0]:
                current_accuracy = correct_predictions / len(results)
                print(f"Current Accuracy: {current_accuracy:.2%}")

                log_file = open(log_path, 'a', newline='')
                log_writer = csv.writer(log_file)
                log_writer.writerow([len(results), current_accuracy])
                log_file.close()

    # Calculate and display final accuracy if answers are available
    if 'answerKey' in data[0]:
        accuracy = correct_predictions / total_questions
        print(f"\nFinal Accuracy: {accuracy:.2%}")

    # Save results
    save_results(results, args.output_file)
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()