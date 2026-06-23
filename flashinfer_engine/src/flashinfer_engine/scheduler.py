"""
Minimal padding-based dynamic batch scheduler.

Collects multiple Request objects, pads them to the same length,
runs a single batched prefill + decode loop, and returns all outputs.

Usage:
    from flashinfer_engine.scheduler import Request, BatchScheduler
    from your_runtime import InferenceEngine

    engine = InferenceEngine("gpt2", backend="sdpa")
    scheduler = BatchScheduler(engine, max_batch_size=8)

    scheduler.add(Request("Hello world", max_new_tokens=30))
    scheduler.add(Request("Once upon a time", max_new_tokens=30))
    outputs = scheduler.run()
    for req, text in zip(scheduler.last_batch, outputs):
        print(text)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


# ------------------------------------------------------------------ #
# Request
# ------------------------------------------------------------------ #

@dataclass
class Request:
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 50

    # filled in after scheduling
    _prompt_ids: list[int] = field(default_factory=list, repr=False)
    _output_ids: list[int] = field(default_factory=list, repr=False)


# ------------------------------------------------------------------ #
# Scheduler
# ------------------------------------------------------------------ #

class BatchScheduler:
    """Padding-based batch scheduler for an InferenceEngine-compatible runtime.

    Limitations (by design - this is a prototype):
    - Static padding: all sequences padded to max prompt length in batch.
    - Decode runs for max(max_new_tokens) steps; finished sequences are
      masked out and don't contribute to the output.
    - No paged KV cache - each request gets its own KV slot.
    """

    def __init__(self, engine, max_batch_size: int = 8):
        """
        Args:
            engine: InferenceEngine-compatible runtime instance
            max_batch_size: maximum requests per batch
        """
        self.engine = engine
        self.max_batch_size = max_batch_size
        self._queue: list[Request] = []
        self.last_batch: list[Request] = []

    def add(self, req: Request) -> None:
        """Enqueue a request."""
        self._queue.append(req)

    def run(self) -> list[str]:
        """
        Drain up to max_batch_size requests from the queue,
        run batched inference, return decoded strings.
        """
        batch = self._queue[: self.max_batch_size]
        self._queue = self._queue[self.max_batch_size :]
        self.last_batch = batch

        if not batch:
            return []

        if len(batch) == 1:
            # fast path: no padding needed
            return [self.engine.generate(
                batch[0].prompt,
                max_new_tokens=batch[0].max_new_tokens,
                temperature=batch[0].temperature,
                top_k=batch[0].top_k,
                verbose=False,
            )]

        return self._run_batched(batch)

    # ---------------------------------------------------------------- #
    # Core batched forward
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def _run_batched(self, batch: list[Request]) -> list[str]:
        engine = self.engine
        tok = engine.tokenizer
        B = len(batch)

        # ---- tokenize ------------------------------------------------
        encoded = [tok.encode(r.prompt) for r in batch]
        prompt_lens = [len(e) for e in encoded]
        max_prompt = max(prompt_lens)

        # left-pad so that real tokens are right-aligned
        pad_id = tok.pad_token_id or tok.eos_token_id
        padded = [
            [pad_id] * (max_prompt - len(e)) + e
            for e in encoded
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device="cuda")  # (B, max_prompt)

        # ---- prefill -------------------------------------------------
        if engine.backend == "flashattn_v0":
            logits, past_kv = engine._forward_custom(input_ids, past_kv=None, current_pos=0)
        else:
            out = engine.model(input_ids, use_cache=True)
            logits, past_kv = out.logits, out.past_key_values

        # ---- sample first token for each request ---------------------
        def _sample(lg: torch.Tensor, temps, top_ks) -> torch.Tensor:
            """lg: (B, vocab) - sample one token per row."""
            results = []
            for i in range(lg.shape[0]):
                row = lg[i] / max(temps[i], 1e-6)
                if top_ks[i] > 0:
                    cutoff = torch.topk(row, top_ks[i]).values[-1]
                    row[row < cutoff] = float("-inf")
                results.append(torch.multinomial(torch.softmax(row, dim=-1), 1))
            return torch.stack(results)  # (B, 1)

        temps  = [r.temperature for r in batch]
        top_ks = [r.top_k for r in batch]

        last_logits = logits[:, -1, :]          # (B, vocab)
        next_toks   = _sample(last_logits, temps, top_ks)  # (B, 1)

        generated   = [[] for _ in range(B)]
        finished    = [False] * B
        max_decode  = max(r.max_new_tokens for r in batch)
        current_pos = max_prompt  # position of the first generated token

        for step in range(max_decode):
            if all(finished):
                break
            if current_pos >= engine.max_seq_len - 1:
                break

            for i in range(B):
                if not finished[i]:
                    tok_id = next_toks[i].item()
                    generated[i].append(tok_id)
                    if (tok_id == tok.eos_token_id or
                            len(generated[i]) >= batch[i].max_new_tokens):
                        finished[i] = True

            if all(finished):
                break

            # decode step: feed the whole batch's last token together
            if engine.backend == "flashattn_v0":
                logits, past_kv = engine._forward_custom(
                    next_toks, past_kv=past_kv, current_pos=current_pos,
                )
            else:
                out = engine.model(next_toks, past_key_values=past_kv, use_cache=True)
                logits, past_kv = out.logits, out.past_key_values

            last_logits = logits[:, -1, :]
            next_toks   = _sample(last_logits, temps, top_ks)
            current_pos += 1

        # ---- decode output strings -----------------------------------
        outputs = []
        for i, req in enumerate(batch):
            all_ids = encoded[i] + generated[i]
            outputs.append(tok.decode(all_ids, skip_special_tokens=True))

        return outputs


# ------------------------------------------------------------------ #
# Throughput helpers
# ------------------------------------------------------------------ #

def run_sequential(engine, requests: list[Request]) -> tuple[list[str], float]:
    """Run requests one by one. Returns (outputs, total_seconds)."""
    outputs = []
    t0 = time.perf_counter()
    for req in requests:
        text = engine.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            verbose=False,
        )
        outputs.append(text)
    return outputs, time.perf_counter() - t0


def run_batched(engine, requests: list[Request],
                max_batch_size: int = 8) -> tuple[list[str], float]:
    """Run requests in batches. Returns (outputs, total_seconds)."""
    scheduler = BatchScheduler(engine, max_batch_size=max_batch_size)
    for req in requests:
        scheduler.add(req)

    outputs = []
    t0 = time.perf_counter()
    while scheduler._queue:
        outputs.extend(scheduler.run())
    return outputs, time.perf_counter() - t0
