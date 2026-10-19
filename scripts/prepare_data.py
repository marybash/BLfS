#!/usr/bin/env python3
"""Tokenise a text file into a uint16 memmap with your tokenizer.

    python scripts/prepare_data.py --vocab tok/vocab.pkl --merges tok/merges.pkl \
        --input data/tinystories_train.txt --output data/train.bin

Writes ids as little-endian uint16 (vocab_size <= 65535). Streams the input, so it works
on a 2 GB file with little memory; expect a few MB/s with a pure-Python encoder.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    tok = Tokenizer.from_files(args.vocab, args.merges, args.special)

    t0, n = time.time(), 0
    with open(args.input, encoding="utf-8") as f, open(args.output, "wb") as out:
        buf = []
        for tid in tok.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= 1_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(out)
                n += len(buf); buf.clear()
                print(f"\r{n/1e6:.1f}M tokens, {time.time()-t0:.0f}s", end="", file=sys.stderr)
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(out); n += len(buf)

    sec = time.time() - t0
    nbytes = os.path.getsize(args.input)
    stats = {"tokens": n, "bytes": nbytes, "seconds": sec,
             "mb_per_s": nbytes / 1e6 / sec, "bytes_per_token": nbytes / n}
    json.dump(stats, open(args.output + ".json", "w"), indent=2)
    print(f"\nwrote {n} tokens to {args.output}: {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
