import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sample a subset of train_filelist.json (JSON Lines)")
    parser.add_argument("--train_json", type=str, required=True, help="Path to original train_filelist.json")
    parser.add_argument("--out_json", type=str, required=True, help="Path to save the sampled train JSON")
    parser.add_argument("--fraction", type=float, default=0.2, help="Fraction of samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 逐行读取 JSON
    data = []
    with open(args.train_json, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # 抽样
    n_total = len(data)
    n_sample = int(n_total * args.fraction)
    random.seed(args.seed)
    sampled_data = random.sample(data, n_sample)

    # 保存为 JSON Lines
    with open(args.out_json, "w") as f:
        for entry in sampled_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Original entries: {n_total}")
    print(f"Sampled entries: {n_sample}")
    print(f"Saved to: {args.out_json}")

if __name__ == "__main__":
    main()
