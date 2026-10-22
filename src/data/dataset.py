"""Dataset + collator over the unified, teacher-scored training examples
(one line per (doc_id, keypoint) tuple; see data/build_index.py and
distill/precompute_teacher_scores.py for how this file is produced).

Lazily indexed by byte offset so the full corpus never has to sit in memory
at once -- the full join can have well over a million (doc, keypoint) rows.
"""
import json

import torch
from torch.utils.data import Dataset


class ScoredExamplesDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self._fh = None
        self.offsets = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[idx])
        line = self._fh.readline()
        return json.loads(line)


def pad_query_ids(tokenizer, text, max_len, mask_token_id):
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    if len(ids) < max_len:
        ids = ids + [mask_token_id] * (max_len - len(ids))
    return ids


class Collator:
    """Builds a padded batch, deduplicating document encoding when multiple
    sampled keypoints in the batch happen to share the same doc_id."""

    def __init__(self, tokenizer, model_cfg):
        self.tokenizer = tokenizer
        self.query_max_len = model_cfg.query_max_len
        self.doc_max_len = model_cfg.doc_max_len
        self.mask_token_id = tokenizer.convert_tokens_to_ids(model_cfg.mask_token)
        if self.mask_token_id is None or self.mask_token_id == tokenizer.unk_token_id:
            raise ValueError(f"mask_token {model_cfg.mask_token!r} is not a known token id")

    def _tok_query(self, text):
        return pad_query_ids(self.tokenizer, text, self.query_max_len, self.mask_token_id)

    def __call__(self, batch):
        # --- deduplicate document encoding by doc_id ---
        doc_id_to_unique_idx = {}
        unique_bodies = []
        doc_index = []
        for ex in batch:
            doc_id = ex["doc_id"]
            if doc_id not in doc_id_to_unique_idx:
                doc_id_to_unique_idx[doc_id] = len(unique_bodies)
                unique_bodies.append(ex["body"])
            doc_index.append(doc_id_to_unique_idx[doc_id])

        doc_enc = self.tokenizer(
            unique_bodies, add_special_tokens=False, truncation=True, max_length=self.doc_max_len,
            padding="max_length", return_tensors="pt",
        )

        direct_ids = torch.tensor([self._tok_query(ex["direct_query"]) for ex in batch], dtype=torch.long)
        indirect_ids = torch.tensor([self._tok_query(ex["indirect_query"]) for ex in batch], dtype=torch.long)

        max_neg = max((len(ex["negative_queries"]) for ex in batch), default=0)
        max_hardneg = max((len(ex["hard_negative_queries"]) for ex in batch), default=0)
        max_neg = max(max_neg, 1)
        max_hardneg = max(max_hardneg, 1)

        B = len(batch)
        neg_ids = torch.full((B, max_neg, self.query_max_len), self.mask_token_id, dtype=torch.long)
        neg_mask = torch.zeros(B, max_neg)
        teacher_neg = torch.zeros(B, max_neg)
        hardneg_ids = torch.full((B, max_hardneg, self.query_max_len), self.mask_token_id, dtype=torch.long)
        hardneg_mask = torch.zeros(B, max_hardneg)
        teacher_hardneg = torch.zeros(B, max_hardneg)

        teacher_direct = torch.zeros(B)
        teacher_indirect = torch.zeros(B)

        for i, ex in enumerate(batch):
            ts = ex["teacher_scores"]
            teacher_direct[i] = ts["direct"]
            teacher_indirect[i] = ts["indirect"]
            for j, q in enumerate(ex["negative_queries"]):
                neg_ids[i, j] = torch.tensor(self._tok_query(q), dtype=torch.long)
                neg_mask[i, j] = 1.0
                teacher_neg[i, j] = ts["negative_queries"][j]
            for j, q in enumerate(ex["hard_negative_queries"]):
                hardneg_ids[i, j] = torch.tensor(self._tok_query(q), dtype=torch.long)
                hardneg_mask[i, j] = 1.0
                teacher_hardneg[i, j] = ts["hard_negative_queries"][j]

        return {
            "doc_input_ids": doc_enc["input_ids"],
            "doc_attention_mask": doc_enc["attention_mask"],
            "doc_index": torch.tensor(doc_index, dtype=torch.long),
            "direct_query_ids": direct_ids,
            "indirect_query_ids": indirect_ids,
            "neg_query_ids": neg_ids,
            "neg_query_mask": neg_mask,
            "hardneg_query_ids": hardneg_ids,
            "hardneg_query_mask": hardneg_mask,
            "teacher_direct": teacher_direct,
            "teacher_indirect": teacher_indirect,
            "teacher_neg": teacher_neg,
            "teacher_hardneg": teacher_hardneg,
        }
