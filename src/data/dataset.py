"""Lazy scored rows, canonical main keypoints, and fixed-size chunk batches."""
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


def _record_keypoint(record, context):
    fields = [name for name in ("main_keypoint", "keypoint") if name in record]
    if not fields:
        raise ValueError(f"{context} needs main_keypoint or keypoint")
    values = [_keypoint_text(record[name], f"{context}.{name}") for name in fields]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{context} has conflicting main_keypoint and keypoint fields")
    return values[0]


def load_main_keypoints(path):
    """Read a JSON doc-id mapping, a record/list of records, or JSONL records."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        raise ValueError(f"Main-keypoint file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    payload.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid keypoint JSON at {path}:{line_number}") from exc
    points = {}

    def add(doc_id, point):
        key = _document_key(doc_id)
        point = _keypoint_text(point, f"Document {key!r} main keypoint")
        if key in points and points[key] != point:
            raise ValueError(f"Conflicting main keypoints for document {key!r}")
        points[key] = point

    if isinstance(payload, dict) and "doc_id" not in payload:
        for doc_id, value in payload.items():
            add(doc_id, _record_keypoint(value, f"Document {doc_id!r}") if isinstance(value, dict) else value)
    else:
        for index, record in enumerate(payload if isinstance(payload, list) else [payload]):
            if not isinstance(record, dict) or "doc_id" not in record:
                raise ValueError(f"Main-keypoint record {index} needs doc_id and keypoint text")
            add(record["doc_id"], _record_keypoint(record, f"Main-keypoint record {index}"))
    if not points:
        raise ValueError(f"No main keypoints found in {path}")
    return points


class ScoredExamplesDataset(Dataset):
    """Index byte offsets and resolve one main keypoint per document.

    A sidecar or explicit main_keypoint field designates the main keypoint.
    Legacy keypoint fields are a fallback only if every row for that document
    agrees across the whole file. Bodies remain lazily loaded.
    """

    def __init__(self, path, main_keypoints_path=None, require_full_document_teacher=False):
        self.path = path
        self._fh = None
        self._fh_pid = None
        self.offsets = []
        self.main_keypoints = load_main_keypoints(main_keypoints_path) if main_keypoints_path else {}
        legacy, ambiguous, body_digests, document_ids = {}, set(), {}, set()
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
                    document_ids.add(key)
                    body = row.get("body")
                    if not isinstance(body, str) or not body.strip():
                        raise ValueError(f"Document {key!r} has an empty or invalid body")
                    digest = hashlib.sha256(body.encode("utf-8")).digest()
                    if key in body_digests and body_digests[key] != digest:
                        raise ValueError(f"Conflicting bodies for document {key!r}")
                    body_digests[key] = digest
                    if "main_keypoint" in row:
                        point = _keypoint_text(row["main_keypoint"], f"Document {key!r} main_keypoint")
                        if key in self.main_keypoints and self.main_keypoints[key] != point:
                            raise ValueError(f"Conflicting main keypoints for document {key!r}")
                        self.main_keypoints[key] = point
                    if isinstance(row.get("keypoint"), str) and row["keypoint"].strip():
                        point = row["keypoint"].strip()
                        if key in legacy and legacy[key] != point:
                            ambiguous.add(key)
                        legacy[key] = point
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"Scored examples file is empty: {path}")
        for key in document_ids:
            if key in self.main_keypoints:
                continue
            if key in ambiguous:
                raise ValueError(
                    f"Document {key!r} has multiple legacy keypoints; provide an explicit "
                    "main_keypoint field or --main-keypoints-path"
                )
            if key not in legacy:
                raise ValueError(f"Document {key!r} has no nonempty main keypoint")
            self.main_keypoints[key] = legacy[key]

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
        row = json.loads(self._fh.readline())
        row["main_keypoint"] = self.main_keypoints[_document_key(row["doc_id"])]
        return row

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
    """Deduplicate documents, chunk every token, and encode full main keypoints."""

    def __init__(self, tokenizer, model_cfg, main_keypoints=None):
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
        self.main_keypoints = {
            _document_key(key): _keypoint_text(value, f"Document {key!r} main keypoint")
            for key, value in (main_keypoints or {}).items()
        }

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
        unique_ids, unique_bodies, unique_points, doc_index = {}, [], [], []
        for row in batch:
            key = _document_key(row.get("doc_id"))
            point = _keypoint_text(
                row.get("main_keypoint", self.main_keypoints.get(key)),
                f"Document {key!r} main_keypoint",
            )
            if key in self.main_keypoints and self.main_keypoints[key] != point:
                raise ValueError(f"Conflicting main keypoints for document {key!r}")
            body = row.get("body")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"Document {key!r} must have a nonempty body")
            if key not in unique_ids:
                unique_ids[key] = len(unique_bodies)
                unique_bodies.append(body)
                unique_points.append(point)
            else:
                idx = unique_ids[key]
                if unique_bodies[idx] != body:
                    raise ValueError(f"Conflicting bodies for document {key!r}")
                if unique_points[idx] != point:
                    raise ValueError(f"Conflicting main keypoints for document {key!r}")
            doc_index.append(unique_ids[key])

        chunk_ids, chunk_masks, chunk_doc_index, keypoint_ids = [], [], [], []
        for doc_idx, (body, point) in enumerate(zip(unique_bodies, unique_points)):
            ids = self._tok_full(body, f"Document {doc_idx}")
            for start in range(0, len(ids), self.doc_chunk_size):
                chunk = ids[start:start + self.doc_chunk_size]
                n_real = len(chunk)
                chunk_ids.append(chunk + [self.pad_token_id] * (self.doc_chunk_size - n_real))
                chunk_masks.append([1] * n_real + [0] * (self.doc_chunk_size - n_real))
                chunk_doc_index.append(doc_idx)
            keypoint_ids.append(self._tok_full(point, f"Document {doc_idx} keypoint"))
        keypoint_tensor = torch.full(
            (len(unique_bodies), max(map(len, keypoint_ids))), self.pad_token_id, dtype=torch.long,
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
