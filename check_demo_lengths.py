"""Check token length distribution of R1 demonstrations in DeepMath-103K."""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

thresholds = [1024, 2048, 4096, 8192, 12288, 16384]
columns = ["r1_solution_1", "r1_solution_2", "r1_solution_3"]

for col in columns:
    lengths = [len(tokenizer.encode(ex[col])) for ex in tqdm(dataset, desc=col)]
    lengths.sort()
    print(f"\n{col}:")
    print(f"  Total examples: {len(lengths)}")
    print(f"  Min: {lengths[0]}, Median: {lengths[len(lengths)//2]}, Max: {lengths[-1]}")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    for t in thresholds:
        count = sum(1 for l in lengths if l <= t)
        print(f"  <= {t:>5} tokens: {count:>6} ({count/len(lengths)*100:.1f}%)")
