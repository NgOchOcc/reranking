"""Qwen3-Reranker-8B teacher: the discriminative Cross-Encoder teacher from
the proposal's Two-Teacher distillation pipeline (§5). Scores (query,
document) pairs using the model's documented yes/no-logit classification
recipe (NOT text generation): the formatted pair is fed through the causal
LM once, and the softmax between the "yes" and "no" token logits at the
final position gives a calibrated relevance probability in (0, 1).

This is used to produce the Margin-MSE targets Delta s* = Teacher(Q+, D) -
Teacher(Q-, D) that the student is distilled against.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX_TEXT = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX_TEXT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class QwenRerankerTeacher:
    def __init__(self, teacher_cfg, device: str = "cuda"):
        self.cfg = teacher_cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.teacher_name, padding_side="left")
        dtype = getattr(torch, teacher_cfg.torch_dtype) if teacher_cfg.torch_dtype else None
        self.model = AutoModelForCausalLM.from_pretrained(teacher_cfg.teacher_name, torch_dtype=dtype).to(device)
        self.model.eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer(PREFIX_TEXT, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tokenizer(SUFFIX_TEXT, add_special_tokens=False)["input_ids"]

    def _process_inputs(self, pairs):
        max_len = self.cfg.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        inputs = self.tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=max_len
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.cfg.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def score_batch(self, queries, documents, instruction: str = None) -> list:
        """queries, documents: equal-length lists of strings. Returns a list
        of P(yes) relevance scores in (0, 1), one per pair."""
        instruction = instruction or self.cfg.instruction
        pairs = [format_instruction(instruction, q, d) for q, d in zip(queries, documents)]
        inputs = self._process_inputs(pairs)
        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
        true_logit = logits[:, self.token_true_id]
        false_logit = logits[:, self.token_false_id]
        stacked = torch.stack([false_logit, true_logit], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()
