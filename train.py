"""Run DPO training on generated pairs."""

import argparse
import os

from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--optim_8bit", action="store_true",
                        help="Use 8-bit AdamW optimizer to save memory")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    if len(dataset) == 0:
        print("No training pairs available. Skipping training.")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "skipped.txt"), "w") as f:
            f.write("No training pairs available.")
        return

    print(f"Training on {len(dataset)} DPO pairs")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="no",  # we save manually at the end
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit" if args.optim_8bit else "adamw_torch",
    )

    # ref_model=None means TRL uses the initial model state as reference
    trainer = DPOTrainer(
        model=args.model_path,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.processing_class.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
