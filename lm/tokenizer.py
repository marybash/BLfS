import regex as re
from collections import Counter
import time
import os
import pickle

PAT = (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

SPECIAL_TOKENS = ["<|endoftext|>"]

def count_pretokens(text: str, special_tokens: list[str]):
    if len(special_tokens) == 0:
        chunks = [text]
    else:
        special_regex = "|".join([re.escape(special) for special in special_tokens])
        chunks = re.split(special_regex, text)

    pretoken_counter = Counter()
    for chunk in chunks:
        pretokens = re.findall(PAT, chunk)

        pretoken_counter.update(pretokens)
        
    return pretoken_counter


def read_input_text(input_path: str):
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    return text


def merge(seq: list[bytes], pair: tuple[bytes, bytes]):
    i = 0
    n = len(seq)
    added_pair = pair[0] + pair[1]
    merged_seq = []
    while i < n - 1:
        if seq[i] == pair[0] and seq[i+1] == pair[1]:
            merged_seq.append(added_pair)
            i += 2
        else:
            merged_seq.append(seq[i])
            i += 1
    if i < n:
        merged_seq.append(seq[i])
    return merged_seq


def train_bpe(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes],  list[tuple[bytes, bytes]]]:
    """
    (input_path, vocab_size, special_tokens) -> (vocab (dict[int, bytes]), merges ( list[tuple[bytes, bytes]])) 
    """

    vocab = {i: bytes([i]) for i in range(256)}

    for i, special_token in enumerate(special_tokens):
        vocab[i+256] = special_token.encode("utf-8")

    text = read_input_text(input_path)
    
    counts = count_pretokens(text, special_tokens) # {str: n}
    seqs = {w: tuple(bytes([b]) for b in w.encode()) for w in counts}
    
    pair_counts = Counter()
    pair_to_words = dict()

    for w, c in counts.items():
        for a, b in zip(seqs[w], seqs[w][1:]):
            pair_counts[(a, b)] += c
            if pair_to_words.get((a, b), None) is None:
                pair_to_words[(a, b)] = set()
            pair_to_words[(a, b)].add(w)

    merges = []

    while len(vocab) < vocab_size:
        if len(pair_counts) == 0:
            break

        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        words_with_pair = pair_to_words.pop(best)

        for w in words_with_pair:
            for a, b in zip(seqs[w], seqs[w][1:]):
                pair_counts[(a, b)] -= counts[w]
                if pair_counts[(a, b)] == 0:
                    pair_counts.pop((a, b))
            seqs[w] = merge(seqs[w], best)
            for a, b in zip(seqs[w], seqs[w][1:]):
                pair_counts[(a, b)] += counts[w]
                if pair_to_words.get((a, b), None) is None:
                    pair_to_words[(a, b)] = set()
                pair_to_words[(a, b)].add(w)

    return vocab, merges


def save_bpe(vocab, merges, dir_path: str = "saves"): 
    os.makedirs(dir_path, exist_ok=True)
    vocab_path = os.path.join(dir_path, 'vocab.pkl')
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    merges_path = os.path.join(dir_path, 'merges.pkl')
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)


def train_and_save_bpe(input_path, vocab_size, special_tokens, dir_path):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_bpe(vocab, merges, dir_path)
    return vocab, merges


def split_on_special_tokens(text, special_tokens):
    if special_tokens is None or len(special_tokens) == 0:
        chunks = [text]
    else:
        special_regex = "(" + "|".join([re.escape(special) for special in sorted(special_tokens, key=len, reverse=True)]) + ")"
        chunks = re.split(special_regex, text)
    return chunks


class Tokenizer():

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens

        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.special_to_id = {special_token: self.token_to_id[special_token.encode("utf-8")]  for special_token in special_tokens}

        self.cache = dict()


    def encode(self, text):
        ids = []
        for chunk in split_on_special_tokens(text, self.special_tokens):
            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk]); continue
            for pre in re.findall(PAT, chunk): # pretokens
                if pre in self.cache:
                    ids.extend(self.cache[pre])
                    continue
                toks = [bytes([b]) for b in pre.encode()]
                for pair in self.merges: # by rank
                    toks = merge(toks, pair)
                pre_ids = [self.token_to_id[t] for t in toks]
                self.cache[pre] = pre_ids
                ids.extend(pre_ids)
        return ids


    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


    def encode_iterable(self, itr):
        for line in itr:
            yield from self.encode(line)


    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
         