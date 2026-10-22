"""Hyperparameter / path configuration for the latent-bottleneck reranker pipeline.

All knobs live here as dataclasses; `Config.from_yaml` loads+overrides from a
YAML file (see configs/default.yaml), and CLI scripts may further override
individual fields via argparse before constructing the model/trainer.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    backbone_name: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 1024
    num_summary_tokens: int = 256
    query_max_len: int = 32
    n_query_adapter_layers: int = 2
    n_query_adapter_heads: int = 8
    n_cross_attn_layers: int = 1
    n_cross_attn_heads: int = 8
    doc_max_len: int = 512
    # Reused as the fixed [MASK] pad slot for the query side (Eq. 1): an
    # existing, already-trained, otherwise-unused Qwen3 special token, so we
    # avoid growing the vocab / resizing the tied embedding matrix.
    mask_token: str = "<|vision_pad|>"


@dataclass
class TeacherConfig:
    teacher_name: str = "Qwen/Qwen3-Reranker-8B"
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    max_length: int = 4096
    torch_dtype: str = "bfloat16"
    batch_size: int = 16


@dataclass
class DataConfig:
    keypoints_dir: str = "../data/msmarco-document/full_run"
    negative_dir: str = "../data/msmarco-document/negative_query"
    unified_path: str = "data/unified_examples.jsonl"
    idf_path: str = "data/idf_table.pt"
    scored_path: str = "data/unified_examples.scored.jsonl"
    neg_per_keypoint: int = 2
    hardneg_per_keypoint: int = 2


@dataclass
class LossConfig:
    lambda_chamfer: float = 1.0
    lambda_consistency: float = 0.5
    lambda_sparsity: float = 0.01
    chamfer_beta: float = 1.0


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    grad_accum_steps: int = 4
    epochs: int = 3
    warmup_ratio: float = 0.05
    bf16: bool = True
    log_every: int = 20
    save_every: int = 1000
    seed: int = 42
    num_workers: int = 2
    max_grad_norm: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for section_name, section_cls in [
            ("model", ModelConfig),
            ("teacher", TeacherConfig),
            ("data", DataConfig),
            ("loss", LossConfig),
            ("train", TrainConfig),
        ]:
            if section_name in raw and raw[section_name]:
                current = getattr(cfg, section_name)
                merged = dataclasses.replace(current, **raw[section_name])
                setattr(cfg, section_name, merged)
        return cfg
