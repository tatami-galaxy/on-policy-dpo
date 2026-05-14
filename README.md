/home/ujan/on-policy-dpo/
├── requirements.txt      # dependencies
├── utils.py              # answer extraction, verification, prompt formatting
├── generate.py           # base + conditioned generation, pair filtering via vLLM
├── train.py              # DPO training via TRL
├── eval.py               # MATH-500 evaluation via vLLM
└── run_loop.py           # orchestrates the iterative loop
How to run

cd /home/ujan/on-policy-dpo
pip install -r requirements.txt

# Full loop (10 rounds)
python run_loop.py --model_name Qwen/Qwen3-1.7B

# Or run individual steps manually:
python generate.py --model_path Qwen/Qwen3-1.7B --output_dir outputs/round_0
python train.py --model_path Qwen/Qwen3-1.7B --dataset_path outputs/round_0/dpo_dataset --output_dir outputs/round_0/model
python eval.py --model_path outputs/round_0/model --output_path outputs/round_0/eval_results.json

# Resume from a specific round
python run_loop.py --model_name Qwen/Qwen3-1.7B --start_round 3
Key design decisions
Subprocess isolation: Each step (generate/train/eval) runs as a separate process so vLLM and the DPO trainer don't compete for GPU memory
DPO prompt = plain question: The conditioning is only used to elicit better generations — the DPO training teaches the model to produce that quality from just the question
Reference model: TRL automatically snapshots the model at the start of each round's training as the reference (no explicit ref_model needed)
Stats tracking: Each round logs gen_stats.json (pair yield, base/cond accuracy) and eval_results.json, with a running summary.json
Things to watch for on early runs
Pair yield rate — if too few pairs are kept (< 50), the DPO signal is weak. If too many (> 500), the model is weak and might need an easier dataset
Conditioned responses referencing the demonstration — e.g. "as shown above" — these would be incoherent paired with the plain question prompt. Worth spot-checking a few samples
Answer extraction failures — the extract_answer util is regex-based; if the model doesn't use \boxed{}, we may miss correct answers. Check gen_stats for suspiciously low accuracy numbers