import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.tokenizer import train_and_save_bpe


def peak_memory_gb():
    try:
        import resource 
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1e9 if sys.platform == "darwin" else 1e6)  
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9  # Windows
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="tok")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    vocab, merges = train_and_save_bpe(args.input, args.vocab_size, args.special, args.out)
    elapsed = time.perf_counter() - t0

    longest_token = max(vocab.values(), key=len)
    print(f"train time:  {elapsed:.1f} s")
    print(f"peak memory: {peak_memory_gb():.2f} GB")
    print(f"vocab size:  {len(vocab)}  (saved to {args.out}/)")
    print(f"longest token: {longest_token!r} ({len(longest_token)} bytes)")
    print(f"first 20 merges: {merges[:20]}")


if __name__ == "__main__":
    main()
