"""Evaluate model on MATH-500."""

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import extract_answer, verify_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Load eval dataset
    print(f"Loading eval dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="test")

    # Detect column names (different datasets use different names)
    if "problem" in dataset.column_names:
        question_col = "problem"
    elif "question" in dataset.column_names:
        question_col = "question"
    else:
        raise ValueError(f"Cannot find question column. Available: {dataset.column_names}")

    if "answer" in dataset.column_names:
        answer_col = "answer"
    elif "final_answer" in dataset.column_names:
        answer_col = "final_answer"
    else:
        raise ValueError(f"Cannot find answer column. Available: {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build prompts
    prompts = []
    for example in dataset:
        messages = [{"role": "user", "content": example[question_col]}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    # Generate
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    print(f"Evaluating on {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    results = []
    for i, example in enumerate(dataset):
        ground_truth = example[answer_col]
        generated = outputs[i].outputs[0].text
        predicted = extract_answer(generated)
        is_correct = verify_answer(predicted, ground_truth)
        if is_correct:
            correct += 1
        results.append({
            "problem": example[question_col],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "generated": generated,
            "correct": is_correct,
        })

    accuracy = correct / len(dataset)
    print(f"\nMATH-500 Accuracy: {correct}/{len(dataset)} = {accuracy * 100:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(
            {"accuracy": accuracy, "correct": correct, "total": len(dataset), "results": results},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
