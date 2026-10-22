"""Batch-runs the document compressor over a random sample of a BEIR
corpus, to sanity-check gate (pi) behavior out-of-domain (BEIR documents
are nothing like the MS MARCO training distribution used for fine-tuning).

Pulls the requested BEIR corpus from its HF Hub mirror (org `BeIR`) rather
than beir's own hardcoded UKP zip URL -- some GPU nodes only have egress to
huggingface.co, not arbitrary hosts, and that mirror is what's reachable
there. Samples --num-samples documents, runs each through
LatentBottleneckReranker.encode_document, and reports per-document gate
(pi) statistics plus the nearest vocabulary term for the single
highest-gated slot (a cheap one-slot flavor of
inspect_bottleneck.py's full nearest-neighbor table, reusing its
rogue-dimension-corrected `_cos`).

Usage:
    python -m src.eval.sample_beir_compress \
        --config configs/default.yaml \
        --checkpoint runs/default/checkpoint-7000.pt \
        --beir-dataset scifact --cache-dir data/beir_hf_cache \
        --num-samples 100 --out-csv runs/default/beir_scifact_stats.csv

Requires the `datasets` library (bundled with transformers/
sentence-transformers, should already be installed). Pick any BEIR set
with an HF Hub mirror via --beir-dataset (e.g. scifact, nfcorpus,
trec-covid, fiqa) -- scifact is the smallest/fastest to pull.
"""
import argparse
import csv
import random

import torch
from transformers import AutoTokenizer

from ..config import Config
from ..model.reranker import LatentBottleneckReranker
from .inspect_bottleneck import _cos, inspect, save_multi_doc_html


def _load_beir_corpus(dataset, cache_dir):
    from datasets import load_dataset

    rows = load_dataset(f"BeIR/{dataset}", "corpus", cache_dir=cache_dir)["corpus"]
    return {row["_id"]: {"title": row.get("title", ""), "text": row["text"]} for row in rows}


def _doc_text(doc):
    title, text = doc.get("title", "").strip(), doc.get("text", "").strip()
    return f"{title}. {text}" if title else text


@torch.no_grad()
def compress_one(model, tokenizer, text, doc_max_len, embed_table):
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=doc_max_len, return_tensors="pt")
    doc_out = model.encode_document(enc["input_ids"], enc["attention_mask"])
    pi = doc_out["pi"][0]
    b_d = doc_out["b_d"][0]

    top_slot = int(pi.argmax().item())
    sim = _cos(b_d[top_slot : top_slot + 1], embed_table)[0]
    top_term = tokenizer.convert_ids_to_tokens(int(sim.argmax().item()))

    vals = pi.tolist()
    n = len(vals)
    return {
        "n_tokens": int(enc["attention_mask"].sum().item()),
        "pi_mean": sum(vals) / n,
        "pi_min": min(vals),
        "pi_max": max(vals),
        "n_slots_gt_0.5": sum(1 for v in vals if v > 0.5),
        "n_slots_gt_0.1": sum(1 for v in vals if v > 0.1),
        "top_slot": top_slot,
        "top_slot_vocab": top_term,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="state_dict .pt to load; random init if omitted")
    ap.add_argument("--beir-dataset", default="scifact")
    ap.add_argument("--cache-dir", default="data/beir_hf_cache", help="HF `datasets` cache dir for the downloaded corpus")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-html", default=None, help="colored-text HTML visualization for --num-visualize of the sampled docs")
    ap.add_argument("--num-visualize", type=int, default=5, help="how many of the sampled docs to render into --out-html")
    ap.add_argument("--viz-top-k", type=int, default=10, help="bottleneck slots shown per doc in --out-html, ranked by gate pi")
    ap.add_argument("--viz-vocab-top-k", type=int, default=3, help="nearest vocab tokens shown per slot in --out-html")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    model = LatentBottleneckReranker(cfg.model)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"[sample_beir] loaded checkpoint {args.checkpoint}")
    else:
        print("[sample_beir] WARNING: no --checkpoint given, using randomly-initialized weights")
    model.eval()
    embed_table = model.backbone.get_input_embeddings().weight

    corpus = _load_beir_corpus(args.beir_dataset, args.cache_dir)
    print(f"[sample_beir] loaded {len(corpus)} docs from beir/{args.beir_dataset}")

    doc_ids = list(corpus.keys())
    random.Random(args.seed).shuffle(doc_ids)
    sample_ids = doc_ids[: args.num_samples]

    rows = []
    for doc_id in sample_ids:
        text = _doc_text(corpus[doc_id])
        if not text:
            continue
        stats = compress_one(model, tokenizer, text, cfg.model.doc_max_len, embed_table)
        stats["doc_id"] = doc_id
        rows.append(stats)
        print(
            f"[sample_beir] {doc_id}: pi_mean={stats['pi_mean']:.3f} "
            f"#>0.5={stats['n_slots_gt_0.5']} top_slot={stats['top_slot']}~{stats['top_slot_vocab']!r}"
        )

    pi_means = [r["pi_mean"] for r in rows]
    n_gt_05 = [r["n_slots_gt_0.5"] for r in rows]
    print(f"\n[sample_beir] === aggregate over {len(rows)} docs ===")
    print(f"[sample_beir] pi_mean: avg={sum(pi_means) / len(pi_means):.4f} min={min(pi_means):.4f} max={max(pi_means):.4f}")
    print(f"[sample_beir] #slots>0.5 per doc: avg={sum(n_gt_05) / len(n_gt_05):.1f} min={min(n_gt_05)} max={max(n_gt_05)}")

    if args.out_csv:
        fieldnames = ["doc_id", "n_tokens", "pi_mean", "pi_min", "pi_max", "n_slots_gt_0.5", "n_slots_gt_0.1", "top_slot", "top_slot_vocab"]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[sample_beir] wrote {args.out_csv}")

    if args.out_html:
        viz_ids = [r["doc_id"] for r in rows[: args.num_visualize]]
        sections = []
        for doc_id in viz_ids:
            text = _doc_text(corpus[doc_id])
            viz_rows, doc_tokens, sim_doc, order, pi = inspect(
                model, tokenizer, text, cfg.model.doc_max_len, args.viz_top_k, args.viz_vocab_top_k
            )
            sections.append((doc_id, doc_tokens, sim_doc, order, pi, viz_rows))
        save_multi_doc_html(args.out_html, sections)
        print(f"[sample_beir] saved {len(sections)}-doc colored-text view -> {args.out_html}")


if __name__ == "__main__":
    main()
