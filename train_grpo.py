"""GRPO training baseline for comparison with iterative on-policy DPO.

Uses the same dataset (DeepMath-103K), model (Qwen3-1.7B), and base prompt format.
"""

import argparse
import re

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from utils import extract_boxed_answer, normalize_answer, format_base_prompt


def accuracy_reward(completions, final_answer, **kwargs):
    """Binary reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, gt in zip(completions, final_answer):
        # Handle conversational format
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion

        predicted = extract_boxed_answer(content)
        pred_norm = normalize_answer(predicted)
        gt_norm = normalize_answer(gt)

        if pred_norm is not None and pred_norm == gt_norm:
            rewards.append(1.0)
        else:
            # Try numeric comparison
            try:
                if pred_norm is not None and abs(float(pred_norm) - float(gt_norm)) < 1e-6:
                    rewards.append(1.0)
                    continue
            except (ValueError, TypeError):
                pass
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward for using \\boxed{} format: 1.0 if present, 0.0 otherwise."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion
        if re.search(r"\\boxed\{", content):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_grpo")
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="zwhe99/DeepMath-103K")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size for faster experiments")
    # Training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # Generation
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Number of completions per prompt (G)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    # vLLM
    parser.add_argument("--use_vllm", action="store_true")
    args = parser.parse_args()

    # Load and format dataset
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Format prompts using the same base prompt as our DPO pipeline
    def format_example(example):
        prompt_content = format_base_prompt(example["question"])
        example["prompt"] = [{"role": "user", "content": prompt_content}]
        return example

    dataset = dataset.map(format_example)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        gradient_checkpointing=True,
        bf16=True,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=0.9 if args.use_vllm else None,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        reward_funcs=[accuracy_reward, format_reward],
        reward_weights=[1.0, 0.1],
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.processing_class.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
