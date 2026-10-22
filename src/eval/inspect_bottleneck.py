"""Interprets what each document-compressor bottleneck vector encodes, by
nearest-neighbor lookup in embedding space.

Two views, both via cosine similarity (gate-invariant: pi_k scales b_d by a
positive scalar, so a pruned-vs-kept slot still shows the same
direction/meaning -- we always use the pre-gate `b_d`, and report the gate
pi_k separately as metadata, rather than `b_d_tilde` which goes to exactly
zero -- and undefined direction -- for a fully pruned slot):

  1. Vocabulary view: nearest token(s) in the backbone's (tied) input
     embedding table -- "which word, anywhere in the vocabulary, does this
     latent vector's direction resemble".
  2. Document view: same comparison, restricted to the input-embedding rows
     of the tokens that actually occur in this document -- "which word IN
     THIS DOCUMENT is this bottleneck slot standing in for". Deliberately
     compares against the raw input embeddings (same space `probe_token_
     vs_top_slots` uses), NOT the post-backbone contextualized hidden
     states h_d: h_d shares b_d's "rogue"/outlier dimensions (both are deep
     in the same transformer's residual stream), which dominate raw cosine
     similarity and collapse every token to ~1.0 against a single slot.
     Raw embeddings don't carry those dims, so this comparison isn't
     inflated the same way -- at the cost of losing h_d's contextualization
     (losses.idf_weighted_chamfer_loss's precision term still argmins over
     h_d directly, since that loss isn't a nearest-neighbor *display* and
     doesn't suffer the same interpretability problem).

Usage:
    python -m src.eval.inspect_bottleneck \
        --config configs/default.yaml --checkpoint runs/default/final_model.pt \
        --doc-id <doc_id> --unified data/unified_examples.jsonl \
        --top-k 10 --out-heatmap runs/default/doc_inspect.png
    # or inspect raw text directly instead of a corpus doc_id:
        --text "some document body text ..."
"""
import argparse
import html
import json

import torch
from transformers import AutoTokenizer

from ..config import Config
from ..model.reranker import LatentBottleneckReranker


def _load_text(args):
    if args.text is not None:
        return args.text
    with open(args.unified) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["doc_id"] == args.doc_id:
                return rec["body"]
    raise ValueError(f"doc_id {args.doc_id!r} not found in {args.unified}")


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)


@torch.no_grad()
def compute_gates(model, tokenizer, text, doc_max_len):
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=doc_max_len, return_tensors="pt")
    doc_out = model.encode_document(enc["input_ids"], enc["attention_mask"])
    return doc_out["pi"][0]  # (M,)


def print_gate_summary(pi):
    vals = pi.tolist()
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)
    print(f"[inspect] num_summary_tokens={n}  mean={sum(vals)/n:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    for thresh in (0.9, 0.5, 0.1, 0.01):
        count = sum(1 for v in vals if v > thresh)
        print(f"[inspect]   #slots with pi > {thresh}: {count}")
    print(f"{'rank':>5} {'slot':>5} {'pi':>8}")
    for rank, slot in enumerate(order):
        print(f"{rank:>5} {slot:>5} {vals[slot]:>8.4f}")


@torch.no_grad()
def inspect(model, tokenizer, text, doc_max_len, top_k, vocab_top_k):
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=doc_max_len, return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    doc_out = model.encode_document(input_ids, attention_mask)

    b_d = doc_out["b_d"][0]  # (M, d) -- gate-invariant direction, see module docstring
    pi = doc_out["pi"][0]  # (M,)
    mask = attention_mask[0].bool()

    n_real = int(mask.sum().item())
    doc_token_ids = input_ids[0, :n_real]
    doc_tokens = tokenizer.convert_ids_to_tokens(doc_token_ids.tolist())

    embed_table = model.backbone.get_input_embeddings().weight  # (V, d), tied with lm_head
    doc_embed = embed_table[doc_token_ids]  # (L, d) -- raw input-embedding rows for this doc's own tokens, not h_d (see module docstring)

    order = torch.argsort(pi, descending=True)[:top_k]
    sim_vocab = _cos(b_d[order], embed_table)  # (K, V)
    sim_doc = _cos(b_d[order], doc_embed)  # (K, L)

    rows = []
    for row_i, slot in enumerate(order.tolist()):
        vocab_scores, vocab_idx = sim_vocab[row_i].topk(vocab_top_k)
        vocab_terms = [
            (tokenizer.convert_ids_to_tokens(int(tid)), round(score.item(), 3))
            for tid, score in zip(vocab_idx, vocab_scores)
        ]
        doc_score, doc_pos = sim_doc[row_i].max(dim=0)
        rows.append({
            "slot": slot,
            "gate": round(pi[slot].item(), 3),
            "vocab_nearest": vocab_terms,
            "doc_nearest_pos": int(doc_pos.item()),
            "doc_nearest_token": doc_tokens[int(doc_pos.item())],
            "doc_nearest_score": round(doc_score.item(), 3),
        })
    return rows, doc_tokens, sim_doc, order, pi


@torch.no_grad()
def compute_norms(model, tokenizer, text, doc_max_len):
    """L2 norm of every compressed bottleneck vector (b_d, pre-gate) vs
    every real document hidden state (h_d) -- a scale check: do the two
    populations even live at comparable magnitudes, or is one systematically
    larger (which would itself skew raw cosine comparisons between them)."""
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=doc_max_len, return_tensors="pt")
    doc_out = model.encode_document(enc["input_ids"], enc["attention_mask"])
    n_real = int(enc["attention_mask"][0].sum().item())

    b_d = doc_out["b_d"][0]  # (M, d)
    h_d = doc_out["h_d"][0][:n_real]  # (L, d)
    return b_d.norm(dim=-1), h_d.norm(dim=-1)


def print_norm_comparison(b_norms, h_norms):
    def _stats(name, t):
        vals = t.tolist()
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        print(f"[inspect] {name:24s} n={n:4d}  mean={mean:8.3f}  std={std:7.3f}  min={min(vals):8.3f}  max={max(vals):8.3f}")

    _stats("b_d (compressed) norm", b_norms)
    _stats("h_d (hidden state) norm", h_norms)
    ratio = (sum(b_norms.tolist()) / len(b_norms)) / (sum(h_norms.tolist()) / len(h_norms))
    print(f"[inspect] mean(b_d norm) / mean(h_d norm) = {ratio:.3f}")


def plot_norm_histogram(path, b_norms, h_norms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Two distinct populations being compared -> categorical colors (identity),
    # not the sequential ramp used elsewhere for magnitude (dataviz skill).
    blue, orange = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(h_norms.tolist(), bins=30, alpha=0.6, color=orange, label=f"h_d (hidden states, n={len(h_norms)})")
    ax.hist(b_norms.tolist(), bins=30, alpha=0.6, color=blue, label=f"b_d (compressed slots, n={len(b_norms)})")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("count")
    ax.set_title("vector norm: compressed bottleneck slots vs document hidden states")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(rows):
    header = f"{'slot':>5} {'gate':>6}  {'nearest vocab token(s)':40} {'nearest doc token':25} {'cos':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        vocab_str = ", ".join(f"{t}({s:.2f})" for t, s in r["vocab_nearest"])
        doc_str = f"{r['doc_nearest_token']!r} @pos{r['doc_nearest_pos']}"
        print(f"{r['slot']:>5} {r['gate']:>6.3f}  {vocab_str:40} {doc_str:25} {r['doc_nearest_score']:>6.3f}")


def save_heatmap(path, doc_tokens, sim_doc, order, pi, max_doc_tokens=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    # Sequential "blue" ramp, light -> dark (dataviz skill reference palette,
    # steps 100/200/350/450/600/700) -- magnitude-only, single hue.
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"],
    )
    highlight = "#eb6834"  # categorical slot 2 (orange) -- per-row argmax marker

    n_tok = min(len(doc_tokens), max_doc_tokens)
    mat = sim_doc[:, :n_tok].cpu().numpy()

    fig_w = max(6, n_tok * 0.28)
    fig_h = max(3, len(order) * 0.32 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"slot {s} (π={pi[s]:.2f})" for s in order.tolist()], fontsize=8)
    ax.set_xticks(range(n_tok))
    ax.set_xticklabels(doc_tokens[:n_tok], rotation=90, fontsize=6)
    ax.set_xlabel("document tokens")
    ax.set_title("bottleneck vector <-> document token cosine similarity")

    for row_i in range(mat.shape[0]):
        col = int(mat[row_i].argmax())
        ax.add_patch(Rectangle((col - 0.5, row_i - 0.5), 1, 1, fill=False, edgecolor=highlight, linewidth=2))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def probe_token_vs_top_slots(model, tokenizer, text, doc_max_len, token_str, top_k):
    """Looks `token_str` up in the backbone's (tied) input embedding table
    and scores it (cosine) against this document's top-`top_k` bottleneck
    slots, ranked by gate pi. Returns (scores, slots, pi_vals) -- same
    shape of info `inspect()` returns per-slot, just for one externally
    chosen token instead of every document token."""
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=doc_max_len, return_tensors="pt")
    doc_out = model.encode_document(enc["input_ids"], enc["attention_mask"])
    pi = doc_out["pi"][0]  # (M,)
    b_d = doc_out["b_d"][0]  # (M, d)

    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) != 1:
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"[inspect] WARNING: {token_str!r} tokenizes into {len(token_ids)} pieces {pieces}, using only the first piece {pieces[0]!r}")
    token_vec = model.backbone.get_input_embeddings().weight[token_ids[0] : token_ids[0] + 1]  # (1, d)

    order = torch.argsort(pi, descending=True)[:top_k]
    scores = _cos(token_vec, b_d[order])[0]  # (K,)
    return scores.tolist(), order.tolist(), pi[order].tolist()


def plot_token_vs_slots(path, token_str, scores, slots, pi_vals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Sequential "blue" ramp for magnitude; orange highlight marks the
    # single best-matching slot (dataviz skill reference palette).
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"],
    )
    highlight = "#eb6834"

    vmin, vmax = min(scores), max(scores)
    span = (vmax - vmin) or 1.0
    colors = [cmap((s - vmin) / span) for s in scores]
    best = max(range(len(scores)), key=lambda i: scores[i])
    colors[best] = highlight

    labels = [f"slot {s}\n(π={p:.2f})" for s, p in zip(slots, pi_vals)]

    fig, ax = plt.subplots(figsize=(max(6, len(scores) * 0.55), 4))
    ax.bar(range(len(scores)), scores, color=colors)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="#898781", linewidth=0.8)
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"token {token_str!r} vs top-{len(scores)} compressed slots")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Validated categorical palette (dataviz skill), fixed order: (light, dark).
# Capped at 8 -- a 9th+ slot never gets a generated hue, it folds into "Other".
_CATEGORICAL = [
    ("#2a78d6", "#3987e5"),  # 1 blue
    ("#eb6834", "#d95926"),  # 2 orange
    ("#1baf7a", "#199e70"),  # 3 aqua
    ("#eda100", "#c98500"),  # 4 yellow
    ("#e87ba4", "#d55181"),  # 5 magenta
    ("#008300", "#008300"),  # 6 green
    ("#6250d6", "#9085e9"),  # 7 violet
    ("#e34948", "#e66767"),  # 8 red
]
_OTHER_GRAY = ("#898781", "#898781")


def _rank_color(rank):
    return _CATEGORICAL[rank] if rank < len(_CATEGORICAL) else _OTHER_GRAY


def _render_doc_section(doc_id, doc_tokens, sim_doc, order, pi, rows):
    """For every real document token, finds its nearest bottleneck slot
    (argmax over the top-k columns of `sim_doc`) and renders the token
    colored by that slot's identity, with the cosine score printed below
    it -- the inverse direction of `save_heatmap`'s per-slot argmax.
    Returns one <section> of HTML; `save_colored_text_html` and
    `save_multi_doc_html` both wrap this in a full page."""
    assign = sim_doc.argmax(dim=0)  # (L,) index into `order`, 0..K-1
    best_score = sim_doc.max(dim=0).values  # (L,)
    vocab_by_slot = {r["slot"]: ", ".join(t for t, _ in r["vocab_nearest"]) for r in rows}

    used_ranks = sorted(set(assign.tolist()))
    legend_html = "".join(
        f'<span class="legend-item" style="--c-light:{light};--c-dark:{dark}">'
        f'<span class="swatch"></span>slot {order[rank].item()} '
        f"(π={pi[order[rank].item()]:.2f}) ~ {html.escape(vocab_by_slot.get(order[rank].item(), '?'))}</span>"
        for rank in used_ranks
        for light, dark in [_rank_color(rank)]
    )

    token_spans = []
    for j, tok in enumerate(doc_tokens):
        rank = int(assign[j].item())
        slot = order[rank].item()
        score = best_score[j].item()
        light, dark = _rank_color(rank)
        tip = f"slot {slot} | gate pi={pi[slot]:.2f} | cos={score:.3f} | vocab~ {vocab_by_slot.get(slot, '?')}"
        token_spans.append(
            f'<span class="tok" style="--c-light:{light};--c-dark:{dark}" title="{html.escape(tip)}">'
            f'<span class="tok-text">{html.escape(tok)}</span>'
            f'<span class="tok-score">{score:.2f}</span>'
            f"</span>"
        )

    rows_html = "".join(
        f"<tr><td>{r['slot']}</td><td>{r['gate']:.3f}</td>"
        f"<td>{html.escape(', '.join(t for t, _ in r['vocab_nearest']))}</td></tr>"
        for r in rows
    )

    heading = f"<h2>{html.escape(str(doc_id))}</h2>" if doc_id is not None else ""
    return f"""
  <section class="doc">
    {heading}
    <div class="legend">{legend_html}</div>
    <div class="text">{''.join(token_spans)}</div>
    <table>
      <tr><th>slot</th><th>gate &pi;</th><th>nearest vocab token(s)</th></tr>
      {rows_html}
    </table>
  </section>"""


def _page_shell(title, body_html):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --text-muted: #898781; --gridline: #e1e0d9;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--surface-1); color: var(--text-primary); padding: 24px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
      --text-muted: #898781; --gridline: #2c2c2a;
    }}
  }}
  h1 {{ font-size: 15px; color: var(--text-secondary); font-weight: 600; margin: 0 0 16px; }}
  h2 {{ font-size: 13px; color: var(--text-primary); font-weight: 600; margin: 0 0 10px; font-family: ui-monospace, monospace; }}
  section.doc {{ margin-bottom: 36px; padding-bottom: 28px; border-bottom: 1px solid var(--gridline); }}
  section.doc:last-child {{ border-bottom: none; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; background: var(--c-light); }}
  @media (prefers-color-scheme: dark) {{ .swatch {{ background: var(--c-dark); }} }}
  .text {{ display: flex; flex-wrap: wrap; gap: 4px 2px; line-height: 1.2; }}
  .tok {{
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 3px 5px 2px; border-radius: 4px; border-bottom: 2px solid var(--c-light);
    background: color-mix(in srgb, var(--c-light) 16%, var(--surface-1));
    cursor: default;
  }}
  @media (prefers-color-scheme: dark) {{
    .tok {{ border-bottom-color: var(--c-dark); background: color-mix(in srgb, var(--c-dark) 22%, var(--surface-1)); }}
  }}
  .tok-text {{ font-size: 13px; }}
  .tok-score {{ font-size: 9px; color: var(--text-muted); font-variant-numeric: tabular-nums; }}
  table {{ margin-top: 24px; font-size: 12px; border-collapse: collapse; color: var(--text-secondary); }}
  td, th {{ padding: 3px 10px; border-bottom: 1px solid var(--gridline); text-align: left; }}
</style></head>
<body class="viz-root">
  <h1>{title}</h1>
  {body_html}
</body></html>"""


def save_colored_text_html(path, doc_tokens, sim_doc, order, pi, rows):
    section = _render_doc_section(None, doc_tokens, sim_doc, order, pi, rows)
    page = _page_shell(
        "bottleneck slot assignment per document token (color = nearest slot, number below = cosine score)", section
    )
    with open(path, "w") as f:
        f.write(page)


def save_multi_doc_html(path, docs):
    """docs: list of (doc_id, doc_tokens, sim_doc, order, pi, rows) tuples,
    e.g. one per sampled document -- renders each as its own section,
    stacked in a single page."""
    body = "\n".join(
        _render_doc_section(doc_id, doc_tokens, sim_doc, order, pi, rows)
        for doc_id, doc_tokens, sim_doc, order, pi, rows in docs
    )
    page = _page_shell(f"bottleneck slot assignment -- {len(docs)} documents", body)
    with open(path, "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="state_dict .pt to load; random init if omitted")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--unified", default=None, help="jsonl to look up --doc-id's body text in")
    ap.add_argument("--text", default=None, help="inspect raw text directly instead of --doc-id")
    ap.add_argument("--top-k", type=int, default=10, help="number of bottleneck slots to show, ranked by gate pi")
    ap.add_argument("--vocab-top-k", type=int, default=3, help="nearest vocab tokens shown per slot")
    ap.add_argument("--out-heatmap", default=None, help="optional PNG path for the doc-token x slot heatmap")
    ap.add_argument("--out-html", default=None, help="optional HTML path for the colored-token-by-slot view")
    ap.add_argument("--pi-only", action="store_true", help="just print gate pi stats for every slot, skip nearest-neighbor lookups/plots")
    ap.add_argument("--probe-token", default=None, help="a word to score (cosine) against the document's top-N compressed slots, plotted as a bar chart")
    ap.add_argument("--probe-top-k", type=int, default=16, help="number of top-gated slots to compare --probe-token against")
    ap.add_argument("--out-token-plot", default=None, help="PNG path for the --probe-token bar chart (default: <probe-token>_vs_slots.png)")
    ap.add_argument("--compare-norms", action="store_true", help="print L2-norm stats for b_d (compressed) vs h_d (hidden states) and skip nearest-neighbor lookups/plots")
    ap.add_argument("--out-norm-plot", default=None, help="optional PNG histogram path for --compare-norms")
    args = ap.parse_args()

    if args.text is None and (args.doc_id is None or args.unified is None):
        ap.error("pass --text, or both --doc-id and --unified")

    cfg = Config.from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    model = LatentBottleneckReranker(cfg.model)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"[inspect] loaded checkpoint {args.checkpoint}")
    else:
        print("[inspect] WARNING: no --checkpoint given, using randomly-initialized weights")
    model.eval()

    text = _load_text(args)

    if args.probe_token:
        scores, slots, pi_vals = probe_token_vs_top_slots(model, tokenizer, text, cfg.model.doc_max_len, args.probe_token, args.probe_top_k)
        out_path = args.out_token_plot or f"{args.probe_token}_vs_slots.png"
        plot_token_vs_slots(out_path, args.probe_token, scores, slots, pi_vals)
        print(f"[inspect] saved token-vs-slots plot -> {out_path}")

    if args.pi_only:
        pi = compute_gates(model, tokenizer, text, cfg.model.doc_max_len)
        print_gate_summary(pi)
        return

    if args.compare_norms:
        b_norms, h_norms = compute_norms(model, tokenizer, text, cfg.model.doc_max_len)
        print_norm_comparison(b_norms, h_norms)
        if args.out_norm_plot:
            plot_norm_histogram(args.out_norm_plot, b_norms, h_norms)
            print(f"[inspect] saved norm histogram -> {args.out_norm_plot}")
        return

    rows, doc_tokens, sim_doc, order, pi = inspect(
        model, tokenizer, text, cfg.model.doc_max_len, args.top_k, args.vocab_top_k,
    )
    print_table(rows)

    if args.out_heatmap:
        save_heatmap(args.out_heatmap, doc_tokens, sim_doc, order, pi)
        print(f"[inspect] saved heatmap -> {args.out_heatmap}")

    if args.out_html:
        save_colored_text_html(args.out_html, doc_tokens, sim_doc, order, pi, rows)
        print(f"[inspect] saved colored-text view -> {args.out_html}")


if __name__ == "__main__":
    main()
