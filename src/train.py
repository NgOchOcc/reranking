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
import dataclasses
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import Config
from .data.dataset import Collator, ScoredExamplesDataset
from .data.idf import IDF_FORMULA, IDF_SCOPE, IDF_VERSION
from .losses import coverage_loss, keypoint_loss, soft_target_bce_loss, sparsity_loss, total_loss
from .model.reranker import LatentBottleneckReranker


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_batch_and_loss(model, batch, idf_table, loss_cfg, accelerator):
    """One wrapped forward includes every trainable branch, including a,b."""
    outputs = model(mode="training", batch=batch)
    doc_out = outputs["doc_out"]
    teacher_pos = torch.stack([batch["teacher_direct"], batch["teacher_indirect"]], dim=1)
    targets = torch.cat([teacher_pos, batch["teacher_neg"], batch["teacher_hardneg"]], dim=1)
    query_mask = torch.cat([
        torch.ones_like(teacher_pos), batch["neg_query_mask"], batch["hardneg_query_mask"],
    ], dim=1)
    bce = soft_target_bce_loss(outputs["logits"], targets, query_mask)
    coverage = coverage_loss(
        doc_out["h_d"], doc_out["doc_attention_mask"], doc_out["b_d"],
        doc_out["chunk_input_ids"], idf_table,
    )
    keypoint = keypoint_loss(
        outputs["keypoint_vectors"], batch["keypoint_attention_mask"],
        doc_out["document_vectors"][batch["keypoint_doc_index"]],
        doc_out["document_summary_mask"][batch["keypoint_doc_index"]],
    )
    sparsity = sparsity_loss(
        doc_out["pi"], doc_out["chunk_doc_index"],
        num_documents=doc_out["document_vectors"].shape[0],
    )
    return total_loss(bce, coverage, keypoint, sparsity, loss_cfg)


def load_idf_table(path, tokenizer_name, vocab_size):
    """Reject old signed/prefix-only IDF artifacts rather than reinterpret them."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or (
        blob.get("version") != IDF_VERSION
        or blob.get("formula") != IDF_FORMULA
        or blob.get("scope") != IDF_SCOPE
    ):
        raise ValueError("IDF table uses an older formula or corpus scope; rebuild it with python -m src.data.idf")
    if blob.get("tokenizer") != tokenizer_name:
        raise ValueError("IDF tokenizer does not match model.backbone_name")
    table = blob["idf"]
    if table.ndim != 1 or table.numel() != vocab_size:
        raise ValueError("IDF table length must match the student tokenizer vocabulary")
    if not torch.isfinite(table).all() or (table < 0).any():
        raise ValueError("IDF weights must be finite and nonnegative")
    return table.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scored-path", default=None, help="override data.scored path from config")
    ap.add_argument("--idf-path", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.scored_path:
        cfg.data.scored_path = args.scored_path
    if args.idf_path:
        cfg.data.idf_path = args.idf_path
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
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
        # The horizon below is in per-process optimizer updates. Disable
        # Accelerate's default world-size multiplier for scheduler.step().
        step_scheduler_with_optimizer=False,
    )
    accelerator.init_trackers("latent_reranker")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    model = LatentBottleneckReranker(cfg.model)

    idf_table = load_idf_table(idf_path, cfg.model.backbone_name, len(tokenizer))

    dataset = ScoredExamplesDataset(scored_path, require_full_document_teacher=True)
    if not len(dataset):
        raise ValueError("Training data contains no scored examples")
    collator = Collator(tokenizer, cfg.model)
    dataloader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collator,
        num_workers=cfg.train.num_workers, drop_last=False,
    )
    # Prepare the dataloader first so len(dataloader) already reflects the
    # per-process shard under multi-GPU -- computing total_steps from the
    # pre-prepare (unsharded) length would overcount by ~num_processes and
    # miscalibrate the warmup/decay schedule.
    dataloader = accelerator.prepare(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = math.ceil(len(dataloader) / cfg.train.grad_accum_steps) * cfg.train.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.train.warmup_ratio), num_training_steps=total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    idf_table = idf_table.to(accelerator.device)
    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "config.yaml"), "w") as config_file:
            yaml.safe_dump(dataclasses.asdict(cfg), config_file, sort_keys=False)
        tokenizer.save_pretrained(output_dir)

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
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
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
