"""Build nonnegative coverage weights over complete original documents.

idf(t) = log((N + 1) / (df(t) + 1)). Document frequency counts each original
document once, independently of its chunks. Ubiquitous tokens have weight
zero and unseen tokens have weight log(N + 1).

Usage:
    python -m src.data.idf \
        --keypoints-dir ../data/msmarco-document/full_run \
        --tokenizer Qwen/Qwen3-Embedding-0.6B \
        --out data/idf_table.pt
"""
import argparse
import glob
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import torch
from transformers import AutoTokenizer

_TOKENIZER = None
IDF_VERSION = 2
IDF_FORMULA = "log((N+1)/(df+1))"
IDF_SCOPE = "full_documents"


def _init_worker(tokenizer_name):
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def _doc_token_set(body):
    ids = _TOKENIZER(body, add_special_tokens=False, truncation=False)["input_ids"]
    return set(ids)


def _iter_bodies(keypoints_dir, limit=None):
    n = 0
    seen = {}
    for path in sorted(glob.glob(os.path.join(keypoints_dir, "keypoints_part*.jsonl"))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = rec.get("body")
                if isinstance(body, str) and body.strip():
                    if rec.get("doc_id") is not None:
                        doc_id = str(rec["doc_id"])
                        digest = hashlib.sha256(body.encode("utf-8")).digest()
                        if doc_id in seen:
                            if seen[doc_id] != digest:
                                raise ValueError(f"Conflicting bodies for document {doc_id!r}")
                            continue
                        seen[doc_id] = digest
                    yield body
                    n += 1
                    if limit is not None and n >= limit:
                        return


def build_idf_table(keypoints_dir, tokenizer_name, vocab_size, num_workers=8, limit=None, chunksize=64):
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied")
    df = Counter()
    n_docs = 0
    bodies_iter = _iter_bodies(keypoints_dir, limit=limit)
    if num_workers <= 1:
        _init_worker(tokenizer_name)
        for body in bodies_iter:
            for tok in _doc_token_set(body):
                df[tok] += 1
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"[idf] processed {n_docs} docs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(tokenizer_name,)) as ex:
            for token_set in ex.map(_doc_token_set, bodies_iter, chunksize=chunksize):
                for tok in token_set:
                    df[tok] += 1
                n_docs += 1
                if n_docs % 5000 == 0:
                    print(f"[idf] processed {n_docs} docs", flush=True)

    if n_docs == 0:
        raise ValueError("Cannot build coverage weights from an empty document corpus")
    idf = torch.full((vocab_size,), fill_value=math.log(n_docs + 1), dtype=torch.float32)
    for tok, count in df.items():
        if not 0 <= tok < vocab_size:
            raise ValueError(f"Token id {tok} lies outside the configured vocabulary")
        idf[tok] = math.log((n_docs + 1) / (1 + count))
    return idf, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints-dir", default="../data/msmarco-document/full_run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="data/idf_table.pt")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap docs processed, for smoke tests")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tok)

    idf, n_docs = build_idf_table(
        args.keypoints_dir, args.tokenizer, vocab_size, num_workers=args.num_workers, limit=args.limit
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "idf": idf,
        "n_docs": n_docs,
        "tokenizer": args.tokenizer,
        "version": IDF_VERSION,
        "formula": IDF_FORMULA,
        "scope": IDF_SCOPE,
    }, args.out)
    print(f"[idf] n_docs={n_docs} vocab_size={vocab_size} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
