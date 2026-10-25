"""Lazy scored query/keypoint rows and complete-document chunk batches."""
from __future__ import annotations

import hashlib
import json
import math
import os
from numbers import Real

import torch
from torch.utils.data import Dataset


def _document_key(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("doc_id must be a nonempty string or integer")
    return str(value)


def _keypoint_text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty keypoint string")
    return value.strip()


class ScoredExamplesDataset(Dataset):
    """Index scored rows without retaining document bodies in memory.

    Each row supplies its own nonempty keypoint. Different keypoints for the
    same document are valid and remain associated with their source rows.
    """

    def __init__(self, path, require_full_document_teacher=False):
        self.path = path
        self._fh = None
        self._fh_pid = None
        self.offsets = []
        body_digests = {}
        teacher_settings = None
        with open(path, "rb") as handle:
            offset = 0
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid scored JSON at {path}:{line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Scored row at {path}:{line_number} must be an object")
                    if require_full_document_teacher:
                        metadata = row.get("teacher_metadata")
                        if not isinstance(metadata, dict) or metadata.get("document_scope") != "full_document":
                            raise ValueError(
                                f"Scored row at {path}:{line_number} has no full-document teacher provenance; "
                                "regenerate legacy prefix targets with python -m "
                                "src.distill.precompute_teacher_scores, or explicitly describe the "
                                "full-document policy used to produce external targets in teacher_metadata."
                            )
                        settings = tuple(metadata.get(field) for field in (
                            "document_scope", "teacher_name", "instruction", "max_length", "policy", "source",
                        ))
                        if teacher_settings is not None and teacher_settings != settings:
                            raise ValueError("Scored rows use inconsistent teacher settings or document policies")
                        teacher_settings = settings
                    key = _document_key(row.get("doc_id"))
                    body = row.get("body")
                    if not isinstance(body, str) or not body.strip():
                        raise ValueError(f"Document {key!r} has an empty or invalid body")
                    digest = hashlib.sha256(body.encode("utf-8")).digest()
                    if key in body_digests and body_digests[key] != digest:
                        raise ValueError(f"Conflicting bodies for document {key!r}")
                    body_digests[key] = digest
                    _keypoint_text(row.get("keypoint"), f"Document {key!r} row keypoint")
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"Scored examples file is empty: {path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.path, "rb")
            self._fh_pid = pid
        self._fh.seek(self.offsets[idx])
        return json.loads(self._fh.readline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state


def pad_query_ids(tokenizer, text, max_len, mask_token_id):
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    return ids + [mask_token_id] * (max_len - len(ids))


def _probability(value, context):
    if not isinstance(value, Real) or not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError(f"{context} must be a finite teacher probability in [0, 1]")
    return float(value)


class Collator:
    """Deduplicate documents and (document, keypoint) pairs independently."""

    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        self.query_max_len = model_cfg.query_max_len
        self.doc_chunk_size = model_cfg.doc_chunk_size
        if self.query_max_len < 1 or self.doc_chunk_size < 1:
            raise ValueError("query_max_len and doc_chunk_size must be positive")
        self.mask_token_id = tokenizer.convert_tokens_to_ids(model_cfg.mask_token)
        if self.mask_token_id is None or self.mask_token_id == tokenizer.unk_token_id:
            raise ValueError(f"mask_token {model_cfg.mask_token!r} is not a known token id")
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id for masked padding")

    def _tok_query(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a nonempty string")
        return pad_query_ids(self.tokenizer, text, self.query_max_len, self.mask_token_id)

    def _tok_full(self, text, context):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context} must be nonempty text")
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            raise ValueError(f"{context} must produce at least one real token")
        return ids

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        unique_ids, unique_bodies, doc_index = {}, [], []
        unique_pairs, unique_points, keypoint_doc_index = set(), [], []
        for row in batch:
            key = _document_key(row.get("doc_id"))
            point = _keypoint_text(row.get("keypoint"), f"Document {key!r} row keypoint")
            body = row.get("body")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"Document {key!r} must have a nonempty body")
            if key not in unique_ids:
                unique_ids[key] = len(unique_bodies)
                unique_bodies.append(body)
            else:
                idx = unique_ids[key]
                if unique_bodies[idx] != body:
                    raise ValueError(f"Conflicting bodies for document {key!r}")
            owner = unique_ids[key]
            doc_index.append(owner)
            pair = (key, point)
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                unique_points.append(point)
                keypoint_doc_index.append(owner)

        chunk_ids, chunk_masks, chunk_doc_index = [], [], []
        for doc_idx, body in enumerate(unique_bodies):
            ids = self._tok_full(body, f"Document {doc_idx}")
            for start in range(0, len(ids), self.doc_chunk_size):
                chunk = ids[start:start + self.doc_chunk_size]
                n_real = len(chunk)
                chunk_ids.append(chunk + [self.pad_token_id] * (self.doc_chunk_size - n_real))
                chunk_masks.append([1] * n_real + [0] * (self.doc_chunk_size - n_real))
                chunk_doc_index.append(doc_idx)
        keypoint_ids = [
            self._tok_full(point, f"Document {owner} keypoint")
            for owner, point in zip(keypoint_doc_index, unique_points)
        ]
        keypoint_tensor = torch.full(
            (len(unique_points), max(map(len, keypoint_ids))), self.pad_token_id, dtype=torch.long,
        )
        keypoint_mask = torch.zeros_like(keypoint_tensor)
        for i, ids in enumerate(keypoint_ids):
            keypoint_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            keypoint_mask[i, :len(ids)] = 1

        negative_lists, hard_lists = [], []
        for row in batch:
            negative, hard = row.get("negative_queries", []), row.get("hard_negative_queries", [])
            if not isinstance(negative, list) or not isinstance(hard, list):
                raise ValueError("Negative query fields must be lists")
            negative_lists.append(negative)
            hard_lists.append(hard)
        max_neg = max(1, max(map(len, negative_lists)))
        max_hardneg = max(1, max(map(len, hard_lists)))
        batch_size = len(batch)
        neg_ids = torch.full((batch_size, max_neg, self.query_max_len), self.mask_token_id, dtype=torch.long)
        neg_mask, teacher_neg = torch.zeros(batch_size, max_neg), torch.zeros(batch_size, max_neg)
        hard_ids = torch.full((batch_size, max_hardneg, self.query_max_len), self.mask_token_id, dtype=torch.long)
        hard_mask, teacher_hard = torch.zeros(batch_size, max_hardneg), torch.zeros(batch_size, max_hardneg)
        teacher_direct, teacher_indirect = torch.zeros(batch_size), torch.zeros(batch_size)
        direct_ids, indirect_ids = [], []
        for i, (row, negatives, hard_negatives) in enumerate(zip(batch, negative_lists, hard_lists)):
            direct_ids.append(self._tok_query(row.get("direct_query")))
            indirect_ids.append(self._tok_query(row.get("indirect_query")))
            scores = row.get("teacher_scores")
            if not isinstance(scores, dict):
                raise ValueError("Each scored row needs a teacher_scores object")
            teacher_direct[i] = _probability(scores.get("direct"), "teacher_scores.direct")
            teacher_indirect[i] = _probability(scores.get("indirect"), "teacher_scores.indirect")
            for queries, field, ids_tensor, mask, targets in (
                (negatives, "negative_queries", neg_ids, neg_mask, teacher_neg),
                (hard_negatives, "hard_negative_queries", hard_ids, hard_mask, teacher_hard),
            ):
                values = scores.get(field, [])
                if not isinstance(values, list) or len(values) != len(queries):
                    raise ValueError(f"teacher_scores.{field} must match query count and ordering")
                for j, (query, value) in enumerate(zip(queries, values)):
                    ids_tensor[i, j] = torch.tensor(self._tok_query(query), dtype=torch.long)
                    mask[i, j] = 1
                    targets[i, j] = _probability(value, f"teacher_scores.{field}[{j}]")

        return {
            "doc_input_ids": torch.tensor(chunk_ids, dtype=torch.long),
            "doc_attention_mask": torch.tensor(chunk_masks, dtype=torch.long),
            "chunk_doc_index": torch.tensor(chunk_doc_index, dtype=torch.long),
            "doc_index": torch.tensor(doc_index, dtype=torch.long),
            "keypoint_doc_index": torch.tensor(keypoint_doc_index, dtype=torch.long),
            "keypoint_input_ids": keypoint_tensor,
            "keypoint_attention_mask": keypoint_mask,
            "direct_query_ids": torch.tensor(direct_ids, dtype=torch.long),
            "indirect_query_ids": torch.tensor(indirect_ids, dtype=torch.long),
            "neg_query_ids": neg_ids,
            "neg_query_mask": neg_mask,
            "hardneg_query_ids": hard_ids,
            "hardneg_query_mask": hard_mask,
            "teacher_direct": teacher_direct,
            "teacher_indirect": teacher_indirect,
            "teacher_neg": teacher_neg,
            "teacher_hardneg": teacher_hard,
        }
