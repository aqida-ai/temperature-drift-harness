"""
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP:
- CORE_LOGIC: signed entropy observations retain their measurement boundary.
- elm_eulerian: causal shifting and target selection are explicit composition.
- n_s_u: held-out next-token evidence is distinct from entropy preservation.
- AQiDA_Physics/Assessment for Genenral AI: cross-entropy is the declared comparator objective.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Finite selected-text
quality metrics; no global quality, public benchmark, or phase-cancellation claim.
FALSIFICATION_TESTS: direct cross-entropy equality, exact label counts, no prompt
or cross-sequence n-gram leakage, unequal-length token-weighted aggregation.
"""
from __future__ import annotations

import math


def make_labels(input_ids, prompt_tokens, policy='continuation_only'):
    """HF labels retain input positions; causal models shift them internally.

    Equal-length, unpadded examples only. The first continuation is at position
    prompt_tokens and is predicted by the final prompt-position logit.
    """
    import torch
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError('Expected an unpadded [batch, tokens] int64 tensor.')
    if input_ids.shape[0] < 1 or not 1 <= prompt_tokens < input_ids.shape[1]:
        raise ValueError('Each example needs a nonempty prompt and continuation.')
    if policy not in {'continuation_only', 'full_sequence'}:
        raise ValueError('Unknown prompt label policy: ' + str(policy))
    labels = input_ids.clone()
    if policy == 'continuation_only':
        labels[:, :prompt_tokens] = -100
    return labels


def count_labels(labels):
    """Count only labels actually predicted after the one-position shift."""
    return int((labels[:, 1:] != -100).sum().item())


def label_budget(examples, prompt_tokens, continuation_tokens, policy):
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
               for v in [examples, prompt_tokens, continuation_tokens]):
        raise ValueError('Budget dimensions must be positive integers.')
    if policy not in {'continuation_only', 'full_sequence'}:
        raise ValueError('Unknown prompt label policy.')
    fixed = examples * (prompt_tokens - 1) if policy == 'full_sequence' else 0
    continuation = examples * continuation_tokens
    return dict(continuation_labels=continuation, fixed_prompt_labels=fixed,
                total_labels=continuation + fixed, prompt_label_policy=policy)


def continuation_log_loss(logits, input_ids, prompt_tokens):
    """Return float64 per-example NLL sums and exact target counts.

    This evaluates raw teacher-forced model predictions with beta=1. No decoder
    temperature, nucleus filtering, chat template, or generation score enters it.
    """
    import torch
    labels = make_labels(input_ids, prompt_tokens)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError('Logits must cover every input position.')
    if input_ids.min().item() < 0 or input_ids.max().item() >= logits.shape[-1]:
        raise ValueError('Token ID outside model vocabulary.')
    selected = logits[:, prompt_tokens - 1:-1, :].double()
    target = labels[:, prompt_tokens:]
    log_probabilities = torch.log_softmax(selected, dim=-1)
    losses = -log_probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(losses).all()):
        raise ValueError('Nonfinite held-out log loss.')
    return [dict(nll_sum=float(row.sum().item()), target_tokens=row.numel())
            for row in losses]


def aggregate_log_loss(rows):
    if not rows or any(r['target_tokens'] <= 0 or
                       not math.isfinite(r['nll_sum']) or r['nll_sum'] < 0 for r in rows):
        raise ValueError('Positive counts and finite nonnegative NLL are required.')
    total = sum(r['target_tokens'] for r in rows)
    nll_sum = math.fsum(r['nll_sum'] for r in rows)
    mean = nll_sum / total
    # Keep the primary NLL when exp overflows; never clip it into a finite success.
    try:
        perplexity = math.exp(mean)
    except OverflowError:
        perplexity = None
    return dict(nll_sum=nll_sum, target_tokens=total, nll_nats_per_token=mean,
                perplexity=perplexity, perplexity_overflow=perplexity is None,
                aggregation='sum of token negative log probabilities / exact target count')


def heldout_quality(model, examples, prompt_tokens, batch_size=1, device='cuda'):
    """Fixed-window quality, not full WikiText perplexity or a leaderboard score."""
    import torch
    if batch_size < 1:
        raise ValueError('Positive batch size required.')
    model.eval()
    model.gradient_checkpointing_disable()
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            ids = torch.as_tensor(examples[start:start + batch_size], dtype=torch.long, device=device)
            out = model(ids, use_cache=False)
            part = continuation_log_loss(out.logits, ids, prompt_tokens)
            rows.extend(dict(example=start + j, **r) for j, r in enumerate(part))
    return dict(**aggregate_log_loss(rows), per_example=rows,
                evaluation='fixed held-out continuations, raw beta=1 model, token-weighted NLL',
                boundary='Within-tokenizer paired quality measurement; base-pretraining overlap unknown. Not full-corpus WikiText perplexity.')


def continuation_distinct_n(sequences, prompt_tokens, n=2):
    """Micro distinct-n over token IDs, never bridging different sequences."""
    if not isinstance(n, int) or n < 1 or prompt_tokens < 0:
        raise ValueError('Invalid n-gram or prompt length.')
    unique = set()
    total = 0
    for sequence in sequences:
        if len(sequence) < prompt_tokens:
            raise ValueError('Sequence is shorter than its declared prompt.')
        continuation = tuple(int(x) for x in sequence[prompt_tokens:])
        for position in range(len(continuation) - n + 1):
            unique.add(continuation[position:position + n])
            total += 1
    return dict(n=n, unique_ngrams=len(unique), total_ngrams=total,
                distinct_n=len(unique) / total if total else None,
                boundary='Token-ID diversity on continuations only; no cross-sequence n-grams and no semantic-quality interpretation.')
