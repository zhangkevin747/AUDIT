"""
Logit Diff Amplification (LDA) for surfacing rare model behaviors.

Amplifies the difference between a base model and a fine-tuned model during
autoregressive generation:

    logits_amplified = logits_ft + α * (logits_ft - logits_base)

This magnifies behavioral changes introduced by fine-tuning, making rare
behaviors (e.g., overcompliance, removed safety guardrails) appear with far
fewer rollouts than standard sampling.

Usage:
    from lda import amplified_generate

    output_ids = amplified_generate(
        base_model, ft_model, input_ids, attention_mask,
        alpha=3.0, temperature=0.7, max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
    )
"""

import torch
from typing import Optional


@torch.no_grad()
def amplified_generate(
    base_model,
    ft_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    alpha: float = 1.0,
    eos_token_id: int = 2,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Autoregressive generation with logit diff amplification.

    Both models must share the same tokenizer/vocabulary and produce logits on
    the same device. Uses KV caching for both models — wall time is roughly
    2x a single model.generate() call, not 2x per token.

    Args:
        base_model:     Reference model (before fine-tuning).
        ft_model:       Target model (after fine-tuning, being evaluated).
        input_ids:      Left-padded tokenized prompts (batch, seq_len).
        attention_mask:  Matching mask for input_ids.
        max_new_tokens: Max tokens to generate per sequence.
        temperature:    Sampling temperature. ≤0 for greedy decoding.
        alpha:          Amplification strength. 0 = normal ft_model sampling.
                        Typical range: 1.0–5.0. High values degrade coherence.
        eos_token_id:   End-of-sequence token ID.
        top_p:          Optional nucleus sampling threshold.

    Returns:
        Token IDs of shape (batch, prompt_len + generated_len).
        Same interface as model.generate().
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # ── Prefill: full prompt through both models ──────────────────────
    base_out = base_model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True
    )
    ft_out = ft_model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=True
    )

    base_cache = base_out.past_key_values
    ft_cache = ft_out.past_key_values

    # First token from amplified logits
    logits_amp = _amplify(base_out.logits[:, -1, :], ft_out.logits[:, -1, :], alpha)
    next_token = _sample(logits_amp, temperature, top_p)  # (batch, 1)

    generated = [next_token]
    finished = next_token.squeeze(-1) == eos_token_id

    # ── Autoregressive decode with KV cache ───────────────────────────
    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        # Extend mask for the newly generated position
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)],
            dim=1,
        )

        base_out = base_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=base_cache,
            use_cache=True,
        )
        ft_out = ft_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=ft_cache,
            use_cache=True,
        )

        base_cache = base_out.past_key_values
        ft_cache = ft_out.past_key_values

        logits_amp = _amplify(base_out.logits[:, -1, :], ft_out.logits[:, -1, :], alpha)
        next_token = _sample(logits_amp, temperature, top_p)

        # Clamp finished sequences to EOS
        next_token[finished] = eos_token_id
        finished = finished | (next_token.squeeze(-1) == eos_token_id)

        generated.append(next_token)

    return torch.cat([input_ids] + generated, dim=1)


def _amplify(
    logits_base: torch.Tensor, logits_ft: torch.Tensor, alpha: float
) -> torch.Tensor:
    """logits_ft + α * (logits_ft – logits_base)"""
    return logits_ft + alpha * (logits_ft - logits_base)


def _sample(
    logits: torch.Tensor, temperature: float, top_p: Optional[float] = None
) -> torch.Tensor:
    """Temperature sampling with optional nucleus (top-p) filtering. Returns (batch, 1)."""
    if temperature <= 1e-8:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Mask everything outside the nucleus, but always keep the top token
        remove = cumprobs <= (1.0 - top_p)
        remove[..., -1:] = False

        # Scatter mask back to original vocabulary order
        remove_orig = torch.zeros_like(logits, dtype=torch.bool)
        remove_orig.scatter_(1, sorted_idx, remove)
        logits[remove_orig] = float("-inf")

    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)