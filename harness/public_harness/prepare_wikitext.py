"""
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP:
- CORE_LOGIC: data provenance and held-out status are scoped observations.
- elm_eulerian: tokenizer, fixed prompts and continuation targets are separate inputs.
- n_s_u: exact source splits, positions and seeds precede model observations.
- Assessment for Genenral AI: held-out quality requires an independent target source.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Deterministic public-text
input preparation. Held out from this adapter loop, not guaranteed unseen in pretraining.
FALSIFICATION_TESTS: pinned files/tokenizer, nonoverlapping source windows, separate
train/test splits, exact integer arrays, no quality measurement or CUDA initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_contains(haystack, needle, item_size=4):
    offset = haystack.find(needle)
    while offset >= 0:
        if offset % item_size == 0:
            return True
        offset = haystack.find(needle, offset + 1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--corpus', default='Salesforce/wikitext')
    parser.add_argument('--corpus-revision', default='b08601e04326c79dfdd32d625aee71d232d685c3')
    parser.add_argument('--prompt-tokens', type=int, default=64)
    parser.add_argument('--continuation-tokens', type=int, default=128)
    parser.add_argument('--contexts', type=int, default=128)
    parser.add_argument('--quality-examples', type=int, default=128)
    parser.add_argument('--train-seed', type=int, default=550155)
    parser.add_argument('--quality-seed', type=int, default=550255)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if any(len(value) != 40 or any(c not in '0123456789abcdef' for c in value)
           for value in [args.model_revision, args.corpus_revision]):
        raise ValueError('Full lowercase source commit hashes are required.')
    if not all(value > 0 for value in [args.prompt_tokens, args.continuation_tokens,
                                       args.contexts, args.quality_examples]):
        raise ValueError('All input dimensions must be positive.')
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / 'inputs.npz', outdir / 'input_manifest.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Prepared inputs are immutable; use a new output directory.')

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision,
                                             local_files_only=True, add_prefix_space=None)
    source_files = {}
    streams = {}
    tokenizer_file = Path(hf_hub_download(args.model, 'tokenizer.json',
                                         revision=args.model_revision, local_files_only=True))
    reference_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    probes = ['', 'a', ' a', '  a', '\na', '\n\na', 'a\tb', 'A punctuation test: 1.04.',
              'def f(x):\n    return x + 1', 'éclair 中文 العربية', '<s> </s> [INST]']
    assert all(tokenizer.encode(text, add_special_tokens=False) ==
               reference_tokenizer.encode(text, add_special_tokens=False).ids for text in probes)
    probe_rows = []
    for split in ['train', 'test']:
        name = 'wikitext-2-raw-v1/' + split + '-00000-of-00001.parquet'
        file = Path(hf_hub_download(args.corpus, name, repo_type='dataset',
                                   revision=args.corpus_revision, local_files_only=True))
        texts = pq.read_table(file, columns=['text']).column('text').to_pylist()
        probe_rows.extend(text for text in texts if text.strip())
        # Match the declared raw-text convention. No chat formatting or BOS/EOS
        # insertion; offsets refer to this joined and tokenized stream.
        joined = '\n\n'.join(text for text in texts if text.strip())
        stream = np.asarray(tokenizer.encode(joined, add_special_tokens=False), dtype='<i4')
        streams[split] = stream
        source_files[split] = dict(repository_file=name, file_sha256=sha(file),
                                  nonempty_rows=sum(bool(text.strip()) for text in texts),
                                  joined_text_sha256=hashlib.sha256(joined.encode('utf-8')).hexdigest(),
                                  tokens=len(stream), token_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest())

    # Falsify whitespace/normalization differences using a fixed source-only
    # sample, before any model prediction or outcome is available.
    sample_indices = np.random.default_rng(550055).choice(len(probe_rows), size=min(128, len(probe_rows)), replace=False)
    probes.extend(probe_rows[int(i)] for i in sample_indices)
    assert all(tokenizer.encode(text, add_special_tokens=False) ==
               reference_tokenizer.encode(text, add_special_tokens=False).ids for text in probes)

    train = streams['train']
    test = streams['test']
    length = args.prompt_tokens + args.continuation_tokens
    starts = np.arange(0, len(train) - args.prompt_tokens + 1, args.prompt_tokens)
    order = np.random.default_rng(args.train_seed).permutation(starts)
    seen = set()
    selected = []
    for start in order:
        key = train[start:start + args.prompt_tokens].tobytes()
        if key not in seen:
            selected.append(int(start))
            seen.add(key)
        if len(selected) == args.contexts:
            break
    assert len(selected) == args.contexts
    train_contexts = np.stack([train[start:start + args.prompt_tokens] for start in selected])

    starts = np.arange(0, len(test) - length + 1, length)
    order = np.random.default_rng(args.quality_seed).permutation(starts)
    train_bytes = train.tobytes()
    quality_offsets = []
    excluded_exact_train_matches = []
    seen = set()
    for start in order:
        key = test[start:start + length].tobytes()
        if aligned_contains(train_bytes, key):
            excluded_exact_train_matches.append(int(start))
            continue
        if key not in seen:
            quality_offsets.append(int(start))
            seen.add(key)
        if len(quality_offsets) == args.quality_examples:
            break
    assert len(quality_offsets) == args.quality_examples
    quality = np.stack([test[start:start + length] for start in quality_offsets])
    for offsets, width in [(selected, args.prompt_tokens), (quality_offsets, length)]:
        ordered = sorted(offsets)
        assert all(b - a >= width for a, b in zip(ordered, ordered[1:]))
    assert train_contexts.shape == (args.contexts, args.prompt_tokens)
    assert quality.shape == (args.quality_examples, length)
    assert source_files['train']['file_sha256'] != source_files['test']['file_sha256']
    assert not torch.cuda.is_initialized()
    np.savez_compressed(paths[0], train_contexts=train_contexts,
                        train_offsets=np.asarray(selected, dtype=np.int64),
                        quality_sequences=quality,
                        quality_offsets=np.asarray(quality_offsets, dtype=np.int64))
    with np.load(paths[0], allow_pickle=False) as saved:
        assert np.array_equal(saved['train_contexts'], train_contexts)
        assert np.array_equal(saved['quality_sequences'], quality)
    result = dict(status='INPUTS_PREPARED_NO_MODEL_MEASUREMENT',
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  model=args.model, model_revision=args.model_revision,
                  corpus=args.corpus, corpus_revision=args.corpus_revision,
                  subset='wikitext-2-raw-v1', source_files=source_files,
                  tokenizer_file_sha256=sha(tokenizer_file), tokenizer_length=len(tokenizer),
                  tokenizer_loading='AutoTokenizer fast backend with add_prefix_space=None to bypass slow conversion in transformers4.50.2; published tokenizer.json normalization retained',
                  tokenizer_reference_check=dict(probe_count=len(probes), exact_token_ID_equality=True,
                                                 probes_sha256=hashlib.sha256(json.dumps(probes,ensure_ascii=False).encode('utf-8')).hexdigest(),
                                                 reference='tokenizers.Tokenizer.from_file(pinned tokenizer.json), add_special_tokens=False'),
                  tokenization='join nonempty rows with two newlines; add_special_tokens=False; no chat template',
                  prompt_tokens=args.prompt_tokens, continuation_tokens=args.continuation_tokens,
                  train_contexts=args.contexts, quality_examples=args.quality_examples,
                  heldout_target_labels=args.quality_examples * args.continuation_tokens,
                  train_seed=args.train_seed, quality_seed=args.quality_seed,
                  sampling='seeded permutation of disjoint source blocks; unique windows; held-out full-window train duplicates excluded before model measurements',
                  train_offsets=selected, quality_offsets=quality_offsets,
                  excluded_exact_train_matches=excluded_exact_train_matches,
                  data_sha256=sha(paths[0]), producer_sha256=sha(Path(__file__)),
                  package_versions={name: importlib.metadata.version(name) for name in
                                    ['numpy', 'torch', 'transformers', 'tokenizers', 'pyarrow', 'huggingface_hub']},
                  CUDA_initialized=torch.cuda.is_initialized(),
                  quality_values_observed=False, S55_protocol_frozen=False,
                  boundary='Quality targets are excluded from this loop\'s training and calibration. Full selected test windows do not appear verbatim in the train stream. Shorter overlap and base-pretraining exposure are not ruled out. Fixed-window perplexity is not full-corpus benchmark perplexity.',
                  public_sources=['https://huggingface.co/datasets/' + args.corpus,
                                  'https://huggingface.co/' + args.model + '/tree/' + args.model_revision])
    paths[1].write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'model', 'model_revision', 'prompt_tokens',
                                          'continuation_tokens', 'train_contexts', 'quality_examples',
                                          'heldout_target_labels', 'data_sha256', 'CUDA_initialized']}, indent=2))


if __name__ == '__main__':
    main()
