"""Main training loop: generate -> train -> eval -> repeat.

Each step runs as a subprocess to ensure clean GPU memory between generation (vLLM)
and training (TRL/transformers).
"""

import argparse
import json
import os
import subprocess
import sys


def run_step(cmd, step_name):
    """Run a command as subprocess, raising on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Iterative on-policy DPO training loop")
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    # Loop
    parser.add_argument("--total_rounds", type=int, default=10)
    parser.add_argument("--start_round", type=int, default=0, help="Resume from this round")
    # Generation
    parser.add_argument("--num_problems", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=16384)
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="zwhe99/DeepMath-103K")
    parser.add_argument("--solution_column", type=str, default="r1_solution_1")
    # Training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--optim_8bit", action="store_true",
                        help="Use 8-bit AdamW optimizer to save memory")
    # Eval
    parser.add_argument("--eval_dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--skip_initial_eval", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    summary = []
    current_model = args.model_name

    # Load existing summary if resuming
    summary_path = os.path.join(args.output_dir, "summary.json")
    if args.start_round > 0 and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        # Find the latest model
        for r in range(args.start_round - 1, -1, -1):
            candidate = os.path.join(args.output_dir, f"round_{r}", "model")
            if os.path.exists(candidate) and not os.path.exists(
                os.path.join(candidate, "skipped.txt")
            ):
                current_model = candidate
                break

    # Initial eval
    if args.start_round == 0 and not args.skip_initial_eval:
        eval_path = os.path.join(args.output_dir, "eval_round_init.json")
        run_step(
            [
                python, os.path.join(script_dir, "eval.py"),
                "--model_path", current_model,
                "--output_path", eval_path,
                "--dataset_name", args.eval_dataset,
                "--max_new_tokens", str(args.max_new_tokens),
            ],
            "INITIAL EVALUATION",
        )
        with open(eval_path) as f:
            init_eval = json.load(f)
        summary.append({"round": "init", "accuracy": init_eval["accuracy"], "pairs": None})
        print(f"\nInitial MATH-500 accuracy: {init_eval['accuracy'] * 100:.1f}%")

    # Main loop
    for round_idx in range(args.start_round, args.total_rounds):
        print(f"\n{'#' * 60}")
        print(f"#  ROUND {round_idx}")
        print(f"{'#' * 60}")

        round_dir = os.path.join(args.output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # 1. Generate pairs
        run_step(
            [
                python, os.path.join(script_dir, "generate.py"),
                "--model_path", current_model,
                "--output_dir", round_dir,
                "--dataset_name", args.dataset_name,
                "--solution_column", args.solution_column,
                "--num_problems", str(args.num_problems),
                "--max_new_tokens", str(args.max_new_tokens),
                "--max_model_len", str(args.max_model_len),
                "--seed", str(42 + round_idx),
            ],
            f"ROUND {round_idx} - GENERATE",
        )

        with open(os.path.join(round_dir, "gen_stats.json")) as f:
            gen_stats = json.load(f)
        print(f"\nPairs kept: {gen_stats['pairs_kept']} / {gen_stats['total']}")

        if gen_stats["pairs_kept"] == 0:
            print("No pairs available. Skipping training for this round.")
            summary.append({"round": round_idx, "accuracy": None, "pairs": 0, "gen_stats": gen_stats})
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            continue

        # 2. Train
        model_output_dir = os.path.join(round_dir, "model")
        dataset_path = os.path.join(round_dir, "dpo_dataset")
        run_step(
            [
                python, os.path.join(script_dir, "train.py"),
                "--model_path", current_model,
                "--dataset_path", dataset_path,
                "--output_dir", model_output_dir,
                "--max_length", str(args.max_model_len),
                "--num_epochs", str(args.num_epochs),
                "--learning_rate", str(args.learning_rate),
                "--beta", str(args.beta),
                "--per_device_batch_size", str(args.per_device_batch_size),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            ] + (["--optim_8bit"] if args.optim_8bit else []),
            f"ROUND {round_idx} - TRAIN",
        )

        current_model = model_output_dir

        # 3. Evaluate
        eval_path = os.path.join(round_dir, "eval_results.json")
        run_step(
            [
                python, os.path.join(script_dir, "eval.py"),
                "--model_path", current_model,
                "--output_path", eval_path,
                "--dataset_name", args.eval_dataset,
                "--max_new_tokens", str(args.max_new_tokens),
            ],
            f"ROUND {round_idx} - EVALUATE",
        )

        with open(eval_path) as f:
            eval_results = json.load(f)
        accuracy = eval_results["accuracy"]
        print(f"\nRound {round_idx} MATH-500 accuracy: {accuracy * 100:.1f}%")

        summary.append({
            "round": round_idx,
            "accuracy": accuracy,
            "pairs": gen_stats["pairs_kept"],
            "gen_stats": gen_stats,
        })

        # Save running summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Final summary
    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    for entry in summary:
        acc = f"{entry['accuracy'] * 100:.1f}%" if entry["accuracy"] is not None else "N/A"
        pairs = entry["pairs"] if entry["pairs"] is not None else "-"
        print(f"  Round {str(entry['round']):>4}: accuracy={acc:>6}  pairs={pairs}")


if __name__ == "__main__":
    main()
