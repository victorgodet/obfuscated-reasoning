"""Token-level cipher utilities for encoding reasoning traces.

Implements a random permutation cipher over vocabulary token IDs,
preserving special/structural tokens so that <think>...</think> tags
and \boxed{} answers remain intact while inner reasoning is encoded.
"""

import json
import random
import re


def create_cipher(tokenizer, seed: int = 42) -> dict[int, int]:
    """Create a random permutation of vocab IDs, excluding special and structural tokens.

    Excluded tokens (mapped to themselves):
    - All special token IDs (bos, eos, pad, unk, all_special_ids)
    - Structural tokens: <think>, </think>, <|im_start|>, <|im_end|>
    """
    # Gather all IDs to exclude
    exclude_ids = set()

    # Special token IDs from tokenizer
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            exclude_ids.add(tid)

    if hasattr(tokenizer, "all_special_ids"):
        exclude_ids.update(tokenizer.all_special_ids)

    # Structural tokens that must remain unchanged
    structural_tokens = ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]
    for tok_str in structural_tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        exclude_ids.update(ids)

    vocab_size = tokenizer.vocab_size
    # IDs eligible for permutation
    permutable = [i for i in range(vocab_size) if i not in exclude_ids]

    rng = random.Random(seed)
    shuffled = permutable.copy()
    rng.shuffle(shuffled)

    cipher = {}
    for orig, perm in zip(permutable, shuffled):
        cipher[orig] = perm

    # Excluded tokens map to themselves (identity)
    for eid in exclude_ids:
        cipher[eid] = eid

    return cipher


def save_cipher(cipher: dict[int, int], path: str) -> None:
    """Persist cipher as JSON with string keys."""
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in cipher.items()}, f)


def load_cipher(path: str) -> dict[int, int]:
    """Load cipher from JSON file."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def invert_cipher(cipher: dict[int, int]) -> dict[int, int]:
    """Create inverse mapping for decoding."""
    return {v: k for k, v in cipher.items()}


def encode_tokens(token_ids: list[int], cipher: dict[int, int]) -> list[int]:
    """Apply permutation cipher to token IDs. Unmapped tokens pass through."""
    return [cipher.get(tid, tid) for tid in token_ids]


def encode_think_content(text: str, tokenizer, cipher: dict[int, int]) -> str:
    """Tokenize text, permute IDs via cipher, decode back to string."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    encoded_ids = encode_tokens(token_ids, cipher)
    return tokenizer.decode(encoded_ids, skip_special_tokens=False)


def encode_response(response: str, tokenizer, cipher: dict[int, int]) -> str:
    """Encode only the content inside <think>...</think>, leaving tags and \\boxed{} untouched."""
    pattern = r"(<think>)(.*?)(</think>)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response

    before = response[: match.start()]
    open_tag = match.group(1)
    think_content = match.group(2)
    close_tag = match.group(3)
    after = response[match.end() :]

    encoded_content = encode_think_content(think_content, tokenizer, cipher)
    return before + open_tag + encoded_content + close_tag + after
