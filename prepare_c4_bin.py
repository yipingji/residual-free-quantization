import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize C4 into a uint32 token .bin file.")
    parser.add_argument("--tokenizer_name", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dataset_name", default="allenai/c4")
    parser.add_argument("--dataset_config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_bin", default="data/c4_train.bin")
    parser.add_argument("--max_tokens", type=int, default=100_000_000)
    parser.add_argument("--buffer_tokens", type=int, default=1_000_000)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_tokens, args.buffer_tokens, args.shuffle_buffer) < 1:
        parser.error("token limits and buffer sizes must be positive")
    return args


def main():
    args = parse_args()
    Path(args.output_bin).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    eos_token = tokenizer.eos_token or ""

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    written = 0
    buffer = []

    with open(args.output_bin, "wb") as f:
        for row in dataset:
            text = row["text"].strip()
            if not text:
                continue

            ids = tokenizer(text + eos_token, verbose=False)["input_ids"]
            remaining = args.max_tokens - written - len(buffer)
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = ids[:remaining]

            buffer.extend(ids)

            if len(buffer) >= args.buffer_tokens:
                chunk = np.asarray(buffer, dtype=np.uint32)
                chunk.tofile(f)
                written += int(chunk.size)
                buffer.clear()
                print(f"wrote {written:,} tokens")

        if buffer:
            chunk = np.asarray(buffer, dtype=np.uint32)
            chunk.tofile(f)
            written += int(chunk.size)

    meta = {
        "tokenizer_name": args.tokenizer_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "dtype": "uint32",
        "tokens": written,
        "bytes": written * 4,
        "output_bin": args.output_bin,
    }
    meta_path = args.output_bin + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"done: {written:,} tokens -> {args.output_bin}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
