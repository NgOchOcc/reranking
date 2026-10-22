"""Training loop CLI for the latent-bottleneck reranker.

Consumes the teacher-scored unified examples produced by
distill/precompute_teacher_scores.py and the IDF table produced by
data/idf.py, and jointly optimizes the (shared) backbone + Residual Query
Adapter + Document Compressor against the four losses in losses.py.

Usage (run on a GPU node; single process or `accelerate launch` for
multi-GPU):
    accelerate launch -m src.train --config configs/default.yaml
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import Config
from .data.dataset import Collator, ScoredExamplesDataset
from .losses import consistency_loss, idf_weighted_chamfer_loss, margin_mse_loss, sparsity_loss, total_loss
from .model.reranker import LatentBottleneckReranker


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_batch_and_loss(model, batch, idf_table, loss_cfg, accelerator):
    # Parameterized paths (encode_document/encode_query) go through the
    # model's own __call__ (mode=...) so that under DistributedDataParallel
    # the gradient-bucket hooks DDP installs on its __call__/forward are
    # actually exercised; direct attribute access on the DDP wrapper would
    # raise AttributeError since custom methods aren't proxied. maxsim has
    # no parameters of its own, so it's called on the unwrapped model.
    raw_model = accelerator.unwrap_model(model)

    doc_out = model(mode="encode_document", input_ids=batch["doc_input_ids"], attention_mask=batch["doc_attention_mask"])
    doc_idx = batch["doc_index"]
    b_tilde_per_ex = doc_out["b_d_tilde"][doc_idx]  # (B, M, d)

    q_direct = model(mode="encode_query", input_ids=batch["direct_query_ids"])
    q_indirect = model(mode="encode_query", input_ids=batch["indirect_query_ids"])
    score_direct = raw_model.maxsim(q_direct, b_tilde_per_ex)
    score_indirect = raw_model.maxsim(q_indirect, b_tilde_per_ex)

    B, K, L = batch["neg_query_ids"].shape
    q_neg = model(mode="encode_query", input_ids=batch["neg_query_ids"].reshape(B * K, L)).reshape(B, K, L, -1)
    score_neg = raw_model.maxsim(q_neg, b_tilde_per_ex.unsqueeze(1).expand(-1, K, -1, -1))

    Bh, Kh, Lh = batch["hardneg_query_ids"].shape
    q_hard = model(mode="encode_query", input_ids=batch["hardneg_query_ids"].reshape(Bh * Kh, Lh)).reshape(Bh, Kh, Lh, -1)
    score_hard = raw_model.maxsim(q_hard, b_tilde_per_ex.unsqueeze(1).expand(-1, Kh, -1, -1))

    pos_scores = torch.stack([score_direct, score_indirect], dim=1)
    pos_mask = torch.ones_like(pos_scores)
    teacher_pos = torch.stack([batch["teacher_direct"], batch["teacher_indirect"]], dim=1)

    neg_scores = torch.cat([score_neg, score_hard], dim=1)
    neg_mask = torch.cat([batch["neg_query_mask"], batch["hardneg_query_mask"]], dim=1)
    teacher_neg = torch.cat([batch["teacher_neg"], batch["teacher_hardneg"]], dim=1)

    margin = margin_mse_loss(pos_scores, neg_scores, pos_mask, neg_mask, teacher_pos, teacher_neg)
    chamfer = idf_weighted_chamfer_loss(
        doc_out["h_d"], batch["doc_attention_mask"], doc_out["b_d"], doc_out["pi"],
        batch["doc_input_ids"], idf_table, beta=loss_cfg.chamfer_beta,
    )
    cons = consistency_loss(score_direct, score_indirect)
    sp = sparsity_loss(doc_out["pi"])
    return total_loss(margin, chamfer, cons, sp, loss_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scored-path", default=None, help="override data.scored path from config")
    ap.add_argument("--idf-path", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    scored_path = args.scored_path or cfg.data.scored_path
    idf_path = args.idf_path or cfg.data.idf_path
    output_dir = args.output_dir or cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)

    set_seed(cfg.train.seed)
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.train.bf16 else "no",
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        log_with="tensorboard",
        project_dir=output_dir,
    )
    accelerator.init_trackers("latent_reranker")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    model = LatentBottleneckReranker(cfg.model)

    idf_blob = torch.load(idf_path, map_location="cpu")
    idf_table = idf_blob["idf"]

    dataset = ScoredExamplesDataset(scored_path)
    collator = Collator(tokenizer, cfg.model)
    dataloader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collator,
        num_workers=cfg.train.num_workers, drop_last=True,
    )
    # Prepare the dataloader first so len(dataloader) already reflects the
    # per-process shard under multi-GPU -- computing total_steps from the
    # pre-prepare (unsharded) length would overcount by ~num_processes and
    # miscalibrate the warmup/decay schedule.
    dataloader = accelerator.prepare(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = (len(dataloader) // cfg.train.grad_accum_steps) * cfg.train.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.train.warmup_ratio), num_training_steps=total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    idf_table = idf_table.to(accelerator.device)

    csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="") if accelerator.is_main_process else None
    csv_writer = None

    pbar = tqdm(total=total_steps, desc="[train]", disable=not accelerator.is_main_process)
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss, logs = forward_batch_and_loss(model, batch, idf_table, cfg.loss, accelerator)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix({k.split("/")[-1]: round(v, 4) for k, v in logs.items()})
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "epoch", "elapsed_s", *logs.keys()])
                        csv_writer.writeheader()
                    row = {"step": global_step, "epoch": epoch, "elapsed_s": round(time.time() - t0, 1), **logs}
                    csv_writer.writerow(row)
                    csv_file.flush()
                    accelerator.log(logs, step=global_step)
                    if global_step % cfg.train.log_every == 0:
                        print(f"[train] step={global_step} epoch={epoch} {logs}", flush=True)
                    if global_step % cfg.train.save_every == 0:
                        ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                        accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                        print(f"[train] saved checkpoint -> {ckpt_path}", flush=True)

    if accelerator.is_main_process:
        pbar.close()
        final_path = os.path.join(output_dir, "final_model.pt")
        accelerator.save(accelerator.unwrap_model(model).state_dict(), final_path)
        print(f"[train] training done, final model -> {final_path}", flush=True)
        if csv_file:
            csv_file.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
