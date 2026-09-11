"""
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP:
- CORE_LOGIC: normalized comparator probabilities feed final signed measurements.
- elm_eulerian: retain the complete finite sampling and optimizer composition.
- n_s_u: explicit seed streams, label budgets and checkpoints back observations.
- AQiDA_Physics: declared cross-entropy objective, without fitted score repairs.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Public Q/V LoRA recursive
training instrument; finite optimization is not a certified I-projection.
FALSIFICATION_TESTS: CPU production-path comparison, prompt-policy loss/gradient
tests, exact checkpoint replay, fixed source bytes and one CUDA process.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import uuid
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('USE_TF', '0')
from . import metrics


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(value):
    print('[' + utc() + '] ' + json.dumps(value, allow_nan=False), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value, exclusive=False):
    path = Path(path)
    raw = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
    if exclusive:
        with path.open('xb') as stream:
            stream.write(raw)
    else:
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_bytes(raw)
        os.replace(temporary, path)


def cuda_pids():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return [int(s.strip()) for s in raw.splitlines() if s.strip()]


def one_cuda_process():
    pids = cuda_pids()
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError('FOREIGN_CUDA_PROCESS_PRESENT: ' + str(pids))
    return pids


def verify_snapshot_files(config, resolver=None):
    """A fixture may supply a local resolver; production resolves only the pinned cache."""
    if 'snapshot_sha256' in config:
        if resolver is None:
            from huggingface_hub import hf_hub_download
            resolver = hf_hub_download
        for name, expected in config['snapshot_sha256'].items():
            cached = Path(resolver(config['model'], name, revision=config['model_revision'], local_files_only=True))
            if cached.stat().st_size != expected['bytes'] or sha(cached) != expected['sha256']:
                raise RuntimeError('PINNED_MODEL_SNAPSHOT_CHANGED: ' + name)


def load_quantized(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    one_cuda_process()
    verify_snapshot_files(config)
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    # This bypasses transformers4.50.2's slow conversion for Mistral. The prepared
    # input manifest independently checks IDs against the official fast backend.
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'],
                                             local_files_only=True, add_prefix_space=None)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['model_revision'],
                                              local_files_only=True, torch_dtype=torch.bfloat16,
                                              quantization_config=quantization,
                                              device_map={'': 0}, low_cpu_mem_usage=True,
                                              attn_implementation='sdpa')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    one_cuda_process()
    return tokenizer, model


def install_lora(model, rank, alpha, device='cuda'):
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(rank, base.in_features, device=device, dtype=torch.float32) * .01)
            self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
            self.scale = alpha / rank

        def forward(self, x):
            base = self.base(x)
            update = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.A), self.B)
            return base + (self.scale * update).to(base.dtype)

    names = []
    for prefix, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if name in ['q_proj', 'v_proj']:
                setattr(module, name, LoRALinear(child))
                names.append(prefix + '.' + name)
    return [p for p in model.parameters() if p.requires_grad], names


def snapshot(parameters):
    return [p.detach().cpu().clone() for p in parameters]


def restore(parameters, weights):
    import torch
    if len(parameters) != len(weights):
        raise ValueError('Checkpoint parameter count changed.')
    with torch.no_grad():
        for parameter, weight in zip(parameters, weights):
            if parameter.shape != weight.shape:
                raise ValueError('Checkpoint parameter shape changed.')
            parameter.copy_(weight)


def save_checkpoint(path, parameters):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(snapshot(parameters), temporary)
    os.replace(temporary, path)


def initialize(parameters, seed):
    import torch
    if len(parameters) % 2:
        raise ValueError('Expected alternating A/B LoRA factors.')
    torch.manual_seed(seed)
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            if index % 2 == 0:
                parameter.normal_(mean=0, std=.01)
            else:
                parameter.zero_()
    return snapshot(parameters)


def distributions(model, contexts, batch_size=4, device='cuda'):
    import torch
    model.eval()
    model.gradient_checkpointing_disable()
    log_probabilities = np.empty((len(contexts), model.config.vocab_size), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            ids = torch.as_tensor(contexts[start:start + batch_size], dtype=torch.long, device=device)
            out = model(ids, use_cache=False)
            values = torch.log_softmax(out.logits[:, -1, :].double(), dim=-1)
            log_probabilities[start:start + len(ids)] = values.cpu().numpy()
    if not np.isfinite(log_probabilities).all():
        raise RuntimeError('NONFINITE_LOG_PROBABILITY')
    probabilities = np.exp(log_probabilities)
    if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12, rtol=0):
        raise RuntimeError('UNNORMALIZED_MEASURED_DISTRIBUTION')
    return probabilities, log_probabilities


def entropy(probabilities, log_probabilities):
    return -np.sum(probabilities * log_probabilities, axis=-1, dtype=np.float64)


def sample(model, contexts, beta, seed, config, device='cuda'):
    import torch
    model.eval()
    model.gradient_checkpointing_disable()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = config['generated_sequences']
    rows = []
    with torch.no_grad():
        for start in range(0, total, config['generation_batch']):
            batch = min(config['generation_batch'], total - start)
            ids = torch.tensor([contexts[(start + j) % len(contexts)] for j in range(batch)],
                               dtype=torch.long, device=device)
            current, past = ids, None
            sequence = [ids]
            for _ in range(config['continuation_tokens']):
                out = model(current, past_key_values=past, use_cache=True)
                past = out.past_key_values
                log_probability = torch.log_softmax(beta * out.logits[:, -1, :].float(), dim=-1)
                current = torch.multinomial(log_probability.exp(), 1, generator=generator)
                sequence.append(current)
            rows.extend(torch.cat(sequence, dim=1).cpu().tolist())
            if start % 128 == 0:
                log(dict(sampled_sequences=start + batch, total=total, beta=beta))
    return np.asarray(rows, dtype=np.int64)


def train(model, parameters, initial, sequences, seed, config, device='cuda'):
    """Reset to the trajectory's initial adapter before every production fit."""
    import torch
    if config['train_batch'] != 1:
        raise ValueError('This instrument declares one sequence per optimizer step.')
    if sequences.shape[1] != config['prompt_tokens'] + config['continuation_tokens']:
        raise ValueError('Training sequence length differs from the frozen design.')
    restore(parameters, initial)
    torch.manual_seed(seed)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    data = torch.from_numpy(sequences)
    schedule = [int(torch.randint(0, len(data), (1,), generator=generator)[0])
                for _ in range(config['production_steps'])]
    trace = []
    consumed = 0
    for step, index in enumerate(schedule, 1):
        ids = data[index:index + 1].to(device)
        labels = metrics.make_labels(ids, config['prompt_tokens'], config['prompt_label_policy'])
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=labels, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('NONFINITE_TRAINING_LOSS')
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
        consumed += metrics.count_labels(labels)
        if step % 50 == 0:
            log(dict(training_step=step, loss=trace[-1]))
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise RuntimeError('NONFINITE_PRODUCTION_ADAPTER')
    budget = metrics.label_budget(len(schedule), config['prompt_tokens'], config['continuation_tokens'],
                                  config['prompt_label_policy'])
    if consumed != budget['total_labels']:
        raise RuntimeError('ACTUAL_LABEL_COUNT_DIFFERS_FROM_DECLARED_BUDGET')
    return dict(loss_trace=trace, training_indices=schedule, consumed_target_labels=consumed,
                label_budget=budget, reset_policy='same per-trajectory initial adapter before every fit',
                boundary='Finite AdamW fit on generated continuations; no projection or Fisher certificate.')
