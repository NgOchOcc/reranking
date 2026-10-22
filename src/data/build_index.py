"""Join keypoints_part*.jsonl + keypoint_queries_part*.jsonl + negative_queries_part*.jsonl
by doc_id into one unified, flat JSONL where each line is a single training
unit: one (doc_id, keypoint) tuple carrying the document body, its direct +
indirect positive queries, and its negative + hard-negative queries.

The three source directories are NOT line-aligned (different doc counts per
part file), so this does a dict-based join keyed by doc_id, and within a
doc, matches keypoints across keypoint_queries/negative_query by exact
keypoint string (both were derived from the same keypoints list upstream).

Usage:
    python -m src.data.build_index \
        --keypoints-dir ../data/msmarco-document/full_run \
        --negative-dir ../data/msmarco-document/negative_query \
        --out data/unified_examples.jsonl
"""
import argparse
import glob
import json
import os


def _load_jsonl_by_doc_id(pattern, limit=None):
    out = {}
    n = 0
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["doc_id"]] = rec
                n += 1
                if limit is not None and n >= limit:
                    return out
    return out


def build_unified_examples(keypoints_dir, negative_dir, limit=None):
    """Yields unified training-unit dicts. Generator so callers can stream to disk."""
    docs = _load_jsonl_by_doc_id(os.path.join(keypoints_dir, "keypoints_part*.jsonl"), limit)
    pos = _load_jsonl_by_doc_id(os.path.join(keypoints_dir, "keypoint_queries_part*.jsonl"), limit)
    neg = _load_jsonl_by_doc_id(os.path.join(negative_dir, "negative_queries_part*.jsonl"), limit)

    doc_ids = sorted(set(docs) & set(pos) & set(neg))
    n_emitted = 0
    n_keypoint_mismatch = 0
    for doc_id in doc_ids:
        body = docs[doc_id].get("body")
        if not body:
            continue
        neg_by_keypoint = {kp["keypoint"]: kp for kp in neg[doc_id].get("keypoints", [])}
        for kp_entry in pos[doc_id].get("keypoints", []):
            keypoint_text = kp_entry["keypoint"]
            queries = {q["query_style"]: q["query"] for q in kp_entry.get("queries", [])}
            direct_q = queries.get("direct")
            indirect_q = queries.get("indirect")
            if not direct_q or not indirect_q:
                continue
            neg_entry = neg_by_keypoint.get(keypoint_text)
            if neg_entry is None:
                n_keypoint_mismatch += 1
                continue
            negative_queries = neg_entry.get("negative_queries", [])
            hard_negative_queries = neg_entry.get("hard_negative_queries", [])
            if not negative_queries and not hard_negative_queries:
                continue
            n_emitted += 1
            yield {
                "doc_id": doc_id,
                "keypoint": keypoint_text,
                "body": body,
                "direct_query": direct_q,
                "indirect_query": indirect_q,
                "negative_queries": negative_queries,
                "hard_negative_queries": hard_negative_queries,
            }
    print(
        f"[build_index] docs joined={len(doc_ids)} examples emitted={n_emitted} "
        f"keypoint_mismatches={n_keypoint_mismatch}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints-dir", default="../data/msmarco-document/full_run")
    ap.add_argument("--negative-dir", default="../data/msmarco-document/negative_query")
    ap.add_argument("--out", default="data/unified_examples.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap docs read per source file, for smoke tests")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for rec in build_unified_examples(args.keypoints_dir, args.negative_dir, limit=args.limit):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[build_index] wrote {n} unified examples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
