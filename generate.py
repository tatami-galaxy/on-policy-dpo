"""Generate base and conditioned completions, filter into DPO pairs."""

import argparse
import json
import os
import random

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import (
    extract_answer,
    format_base_prompt,
    format_conditioned_prompt,
    verify_answer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="zwhe99/DeepMath-103K")
    parser.add_argument("--solution_column", type=str, default="r1_solution_1")
    parser.add_argument("--num_problems", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and sample
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")
    indices = random.sample(range(len(dataset)), min(args.num_problems, len(dataset)))
    sampled = dataset.select(indices)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build prompts and filter by token length
    max_prompt_tokens = args.max_model_len - args.max_new_tokens
    base_prompts = []
    conditioned_prompts = []
    kept_indices = []  # track which examples survive length filtering

    for idx, example in enumerate(sampled):
        question = example["question"]
        demonstration = example[args.solution_column]

        base_content = format_base_prompt(question)
        cond_content = format_conditioned_prompt(question, demonstration)

        base_messages = [{"role": "user", "content": base_content}]
        cond_messages = [{"role": "user", "content": cond_content}]

        base_text = tokenizer.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True
        )
        cond_text = tokenizer.apply_chat_template(
            cond_messages, tokenize=False, add_generation_prompt=True
        )

        # Check token lengths — both prompts must fit within budget
        base_len = len(tokenizer.encode(base_text))
        cond_len = len(tokenizer.encode(cond_text))

        if base_len > max_prompt_tokens or cond_len > max_prompt_tokens:
            continue

        base_prompts.append(base_text)
        conditioned_prompts.append(cond_text)
        kept_indices.append(idx)

    filtered_out = len(sampled) - len(kept_indices)
    print(f"Filtered {filtered_out}/{len(sampled)} problems due to prompt length "
          f"(max_prompt_tokens={max_prompt_tokens})")

    # Generate with vLLM
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
    )

    print(f"Generating {len(base_prompts)} base completions...")
    base_outputs = llm.generate(base_prompts, sampling_params)

    print(f"Generating {len(conditioned_prompts)} conditioned completions...")
    cond_outputs = llm.generate(conditioned_prompts, sampling_params)

    # Filter pairs: keep only where base is wrong and conditioned is correct
    dpo_data = {"prompt": [], "chosen": [], "rejected": []}
    kept_examples = sampled.select(kept_indices)
    stats = {
        "total": len(sampled),
        "filtered_by_length": filtered_out,
        "after_length_filter": len(kept_examples),
        "base_correct": 0,
        "cond_correct": 0,
        "pairs_kept": 0,
        "both_wrong": 0,
        "both_correct": 0,
    }

    for i, example in enumerate(kept_examples):
        ground_truth = example["final_answer"]
        base_text = base_outputs[i].outputs[0].text
        cond_text = cond_outputs[i].outputs[0].text

        base_answer = extract_answer(base_text)
        cond_answer = extract_answer(cond_text)

        base_correct = verify_answer(base_answer, ground_truth)
        cond_correct = verify_answer(cond_answer, ground_truth)

        if base_correct:
            stats["base_correct"] += 1
        if cond_correct:
            stats["cond_correct"] += 1
        if base_correct and cond_correct:
            stats["both_correct"] += 1
        if not base_correct and not cond_correct:
            stats["both_wrong"] += 1

        # Only keep pairs with correctness gap: base wrong, conditioned correct
        if not base_correct and cond_correct:
            # DPO prompt is the plain question (no conditioning).
            # The model learns to produce conditioned-quality output from just the question.
            question = example["question"]
            prompt_messages = [{"role": "user", "content": question}]
            dpo_data["prompt"].append(prompt_messages)
            dpo_data["chosen"].append([{"role": "assistant", "content": cond_text}])
            dpo_data["rejected"].append([{"role": "assistant", "content": base_text}])
            stats["pairs_kept"] += 1

    # Save dataset and stats
    dpo_dataset = Dataset.from_dict(dpo_data)
    dataset_path = os.path.join(args.output_dir, "dpo_dataset")
    dpo_dataset.save_to_disk(dataset_path)

    stats_path = os.path.join(args.output_dir, "gen_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    n_after = stats["after_length_filter"]
    print(f"\nGeneration stats:")
    print(f"  Total sampled:       {stats['total']}")
    print(f"  Filtered by length:  {stats['filtered_by_length']}")
    print(f"  After length filter: {n_after}")
    if n_after > 0:
        print(f"  Base correct:   {stats['base_correct']} ({stats['base_correct']/n_after*100:.1f}%)")
        print(f"  Cond correct:   {stats['cond_correct']} ({stats['cond_correct']/n_after*100:.1f}%)")
        print(f"  Both correct:   {stats['both_correct']}")
        print(f"  Both wrong:     {stats['both_wrong']}")
        print(f"  Pairs kept:     {stats['pairs_kept']} ({stats['pairs_kept']/n_after*100:.1f}%)")
    else:
        print(f"  All problems filtered out by length!")


if __name__ == "__main__":
    main()
