"""
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP:
- CORE_LOGIC: retain signed observations and finite partial-run status.
- elm_eulerian: sampling, reset fitting and quality evaluation remain explicit.
- n_s_u: committed prearm inputs and every stage/checkpoint back the measurements.
- AQiDA_Physics/Assessment for Genenral AI: report entropy and held-out loss separately.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Recursive Q/V LoRA
measurement runner; no automatic statistical, collapse, Fisher or public verdict.
FALSIFICATION_TESTS: exact source/seed binding, base and checkpoint replay,
hash-checked resumable stages, target counts, teacher chain and no-training control.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

from . import precommitment as P


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_runtime_config(root, record):
    config = record['design']['config']
    for key in ['contexts', 'generated_sequences', 'generation_batch', 'production_steps',
                'train_batch', 'rank', 'alpha', 'adapter_modules', 'trainable_parameters',
                'cpu_threads', 'minimum_free_bytes_prelaunch', 'minimum_free_bytes_generation']:
        P.require(P.positive_integer(config[key]), 'Positive integer runtime setting required: ' + key)
    for key in ['learning_rate', 'base_replay_tolerance', 'checkpoint_replay_tolerance']:
        P.require(isinstance(config[key], (float, int)) and math.isfinite(config[key]) and config[key] > 0,
                  'Positive finite runtime setting required: ' + key)
    P.require(config['train_batch'] == 1, 'This production engine supports one sequence per training step.')
    P.require(config['generated_sequences'] % config['contexts'] == 0, 'Every context must receive equal generated sequence counts.')
    P.require(config['generated_sequences'] >= config['contexts'], 'Every context must be sampled.')
    P.require(config['prompt_label_policy'] == 'continuation_only', 'This new runner declares continuation-only supervision.')
    P.require(all(not j.get('fresh', False) for j in record['design']['jobs']), 'This replication runner does not implement a fresh training arm.')
    required = [Path(__file__), Path(__file__).with_name('engine.py'),
                Path(__file__).with_name('metrics.py'), Path(__file__).with_name('precommitment.py')]
    names = [p.resolve().relative_to(root.resolve()).as_posix() for p in required]
    names += [config['input_file'], config['input_manifest'], config['base_conditionals']]
    P.require(set(names) <= set(record['source_files']), 'Runtime code and prepared/base inputs must all be frozen.')
    for name in names:
        P.inside(root, name)
    P.require(platform.python_version() == config['expected_python_version'], 'Python version differs from the commitment.')
    P.require({'numpy', 'torch', 'transformers', 'tokenizers', 'bitsandbytes', 'accelerate', 'huggingface_hub'} <=
              set(config['expected_package_versions']), 'All runtime library versions must be pinned.')
    for name, version in config['expected_package_versions'].items():
        P.require(importlib.metadata.version(name) == version, 'Runtime package differs from the commitment: ' + name)
    return config


def save_arrays(path, **arrays):
    import numpy as np
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def run_job(model, parameters, contexts, quality_examples, base, base_quality,
            job, config, commitment_sha, output, device='cuda', resource_check=None):
    """CPU device is for software fixtures; the command-line runner requires CUDA."""
    import numpy as np
    import torch
    from . import engine as E
    from . import metrics as M
    resource_check = resource_check or E.one_cuda_process
    folder = output / job['name']
    folder.mkdir(parents=True, exist_ok=True)
    trajectory_path = folder / 'trajectory.json'
    if trajectory_path.exists():
        record = read(trajectory_path)
        P.require(record['precommitment_sha256'] == commitment_sha and record['job'] == job,
                  'A resumed trajectory has a different protocol or seed grid.')
    else:
        record = dict(status='RUNNING', started_utc=E.utc(), job=job,
                      precommitment_sha256=commitment_sha, generations=[],
                      scientific_verdict_authorized=False)
    initial_path = folder / 'initial_adapter.pt'
    initial_manifest = folder / 'initial_adapter.json'
    if initial_manifest.exists():
        metadata = read(initial_manifest)
        P.require(metadata['seed'] == job['initialization_seed'] and metadata['sha256'] == E.sha(initial_path),
                  'Initial adapter differs from its manifest.')
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    else:
        P.require(not initial_path.exists(), 'Initial checkpoint has no manifest; preserve and audit before recovery.')
        initial = E.initialize(parameters, job['initialization_seed'])
        E.save_checkpoint(initial_path, parameters)
        E.save_json(initial_manifest, dict(seed=job['initialization_seed'], sha256=E.sha(initial_path)), exclusive=True)
    E.restore(parameters, initial)
    baseP, baseLP = base
    sourceP, sourceLP = base
    start_generation = len(record['generations'])
    P.require(start_generation <= config['generations'], 'Trajectory exceeds its frozen horizon.')
    for index, completed in enumerate(record['generations'], 1):
        saved = folder / f'g{index:02d}'
        P.require(completed['generation'] == index, 'Saved generation grid is not contiguous.')
        for filename, field in [('tokens.npz', 'tokens_sha256'), ('conditionals.npz', 'conditionals_sha256')]:
            P.require(E.sha(saved / filename) == completed[field], 'Completed-generation evidence changed: ' + filename)
        if job['train']:
            for filename, field in [('adapter.pt', 'adapter_sha256'), ('training.json', 'training_sha256')]:
                P.require(E.sha(saved / filename) == completed[field], 'Completed-training evidence changed: ' + filename)
    if start_generation:
        prior = record['generations'][-1]
        prior_folder = folder / f'g{start_generation:02d}'
        if job['train']:
            checkpoint = prior_folder / 'adapter.pt'
            P.require(E.sha(checkpoint) == prior['adapter_sha256'], 'Saved teacher checkpoint changed.')
            E.restore(parameters, torch.load(checkpoint, map_location='cpu', weights_only=True))
            sourceP, sourceLP = E.distributions(model, contexts, device=device)
        expected_file = prior_folder / 'conditionals.npz'
        P.require(E.sha(expected_file) == prior['conditionals_sha256'], 'Saved conditionals changed.')
        with np.load(expected_file, allow_pickle=False) as expected:
            error = float(np.max(np.abs(sourceLP - expected['log_probabilities'])))
        P.require(error <= config['checkpoint_replay_tolerance'], 'CHECKPOINT_LOG_PROBABILITY_REPLAY_FAILED')
        record.setdefault('resumes', []).append(dict(utc=E.utc(), after_generation=start_generation,
                                                     log_probability_max_error=error))
    record['status'] = 'RUNNING'
    E.save_json(trajectory_path, record)
    for index in range(start_generation, config['generations']):
        resource_check()
        P.require(shutil.disk_usage(output).free >= config['minimum_free_bytes_generation'], 'Disk guard requires a resource pause.')
        started = time.monotonic()
        number = index + 1
        generation_folder = folder / f'g{number:02d}'
        generation_folder.mkdir(exist_ok=True)
        stage_path = generation_folder / 'stage.json'
        teacher = E.sha(folder / f'g{index:02d}' / 'adapter.pt') if index and job['train'] else E.sha(initial_path)
        binding = dict(generation=number, precommitment_sha256=commitment_sha,
                       teacher_adapter_sha256=teacher, sampling_seed=job['sampling_seeds'][index],
                       training_seed=job['training_seeds'][index])
        stage = read(stage_path) if stage_path.exists() else dict(**binding, status='STARTED', started_utc=E.utc())
        P.require(all(stage[k] == v for k, v in binding.items()), 'Generation stage belongs to a different teacher or seed.')
        E.save_json(stage_path, stage)
        E.log(dict(trajectory=job['name'], generation=number, beta=job['beta']))
        tokens_path = generation_folder / 'tokens.npz'
        if tokens_path.exists():
            P.require(stage.get('tokens_sha256') == E.sha(tokens_path), 'Samples have no matching saved stage hash.')
            with np.load(tokens_path, allow_pickle=False) as stored:
                sequences = stored['tokens']
        else:
            clock = time.monotonic()
            sequences = E.sample(model, contexts, job['beta'], job['sampling_seeds'][index], config, device=device)
            save_arrays(tokens_path, tokens=sequences)
            stage.update(tokens_sha256=E.sha(tokens_path), sampling_seconds=time.monotonic() - clock,
                         status='SAMPLES_SAVED')
            E.save_json(stage_path, stage)
        P.require(sequences.shape == (config['generated_sequences'], config['prompt_tokens'] + config['continuation_tokens']),
                  'Generated token shape differs from the commitment.')
        expected_prompts = np.asarray([contexts[i % len(contexts)] for i in range(len(sequences))])
        P.require(np.array_equal(sequences[:, :config['prompt_tokens']], expected_prompts), 'Generated prompt cycle changed.')
        P.require(int(sequences.min()) >= 0 and int(sequences.max()) < config['vocabulary'], 'Generated token out of vocabulary.')
        resource_check()
        training_path = generation_folder / 'training.json'
        checkpoint = generation_folder / 'adapter.pt'
        if job['train']:
            if training_path.exists():
                training = read(training_path)
                P.require(training['tokens_sha256'] == stage['tokens_sha256'] and
                          training['precommitment_sha256'] == commitment_sha and
                          training['adapter_sha256'] == E.sha(checkpoint), 'Training/checkpoint binding changed.')
                E.restore(parameters, torch.load(checkpoint, map_location='cpu', weights_only=True))
            else:
                P.require(not checkpoint.exists(), 'Adapter checkpoint lacks a training manifest; preserve and audit before recovery.')
                clock = time.monotonic()
                training = E.train(model, parameters, initial, sequences, job['training_seeds'][index], config, device=device)
                E.save_checkpoint(checkpoint, parameters)
                training.update(tokens_sha256=stage['tokens_sha256'], precommitment_sha256=commitment_sha,
                                adapter_sha256=E.sha(checkpoint), training_seconds=time.monotonic() - clock)
                E.save_json(training_path, training, exclusive=True)
            expected_budget = config['production_steps'] * config['continuation_tokens']
            P.require(training['consumed_target_labels'] == expected_budget, 'Optimizer target exposure differs from the protocol.')
            targetP, targetLP = E.distributions(model, contexts, device=device)
            quality = M.heldout_quality(model, quality_examples, config['prompt_tokens'], device=device)
            adapter_sha = training['adapter_sha256']
        else:
            P.require(all(torch.equal(p.detach().cpu(), w) for p, w in zip(parameters, initial)), 'No-training control changed parameters.')
            targetP, targetLP = base
            quality = base_quality
            training = dict(consumed_target_labels=0, training_seconds=0.)
            adapter_sha = E.sha(initial_path)
        resource_check()
        h0 = E.entropy(sourceP, sourceLP)
        h1 = E.entropy(targetP, targetLP)
        decodedLP = job['beta'] * sourceLP
        decodedLP -= np.logaddexp.reduce(decodedLP, axis=1, keepdims=True)
        decodedH = E.entropy(np.exp(decodedLP), decodedLP)
        drift = h0 - h1
        price = h0 - decodedH
        if index:
            P.require(np.max(np.abs(h0 - np.asarray(record['generations'][-1]['H_production']))) < 1e-10,
                      'Teacher entropy chain changed.')
        conditionals_path = generation_folder / 'conditionals.npz'
        save_arrays(conditionals_path, log_probabilities=targetLP)
        result = dict(generation=number, completed_utc=E.utc(), seconds=time.monotonic() - started,
                      timing_scope='seconds covers this execution attempt; persisted sampling/training phases retain earlier completed work after a resume',
                      sampling_seconds=stage['sampling_seconds'], training_seconds=training['training_seconds'],
                      H_source=h0.tolist(), H_production=h1.tolist(), H_decoded=decodedH.tolist(),
                      D_H=drift.tolist(), J_H=price.tolist(), signed_budget=(price - drift).tolist(),
                      mean_D_H=float(np.mean(drift)), mean_H=float(np.mean(h1)),
                      quality=quality,
                      quality_NLL_change_from_base=quality['nll_nats_per_token'] - base_quality['nll_nats_per_token'],
                      generated_distinct_2=M.continuation_distinct_n(sequences, config['prompt_tokens'], 2),
                      training_target_labels=training['consumed_target_labels'],
                      tokens_sha256=E.sha(tokens_path), adapter_sha256=adapter_sha,
                      training_sha256=E.sha(training_path) if job['train'] else None,
                      conditionals_sha256=E.sha(conditionals_path),
                      scope='One dependent generation. Statistical uncertainty belongs to independent complete seed trajectories.',
                      scientific_verdict_authorized=False)
        record['generations'].append(result)
        E.save_json(trajectory_path, record)
        stage.update(status='MEASURED', completed_utc=result['completed_utc'])
        E.save_json(stage_path, stage)
        E.log(dict(trajectory=job['name'], generation=number, D_H=result['mean_D_H'],
                   H=result['mean_H'], heldout_NLL=quality['nll_nats_per_token'], seconds=result['seconds']))
        sourceP, sourceLP = targetP, targetLP
        if (output / 'PAUSE_AFTER_GENERATION').exists():
            record['status'] = 'RESOURCE_PAUSE'
            E.save_json(trajectory_path, record)
            return False
    record['status'] = 'COMPLETE_TRAJECTORY_AWAITS_ANALYSIS'
    record['completed_utc'] = E.utc()
    E.save_json(trajectory_path, record)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--precommitment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-packages', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.runtime_packages:
        sys.path.insert(0, str(args.runtime_packages.resolve()))
    root = args.root.resolve()
    verified = P.verify(root, args.precommitment, require_committed=True)
    record = read(args.precommitment)
    config = validate_runtime_config(root, record)
    if args.check_only:
        print(json.dumps(dict(status='PASS_CPU_RUN_CONFIGURATION', commitment=verified), indent=2))
        return
    from . import engine as E
    from . import metrics as M
    import numpy as np
    import torch
    E.one_cuda_process()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    P.require(shutil.disk_usage(output).free >= config['minimum_free_bytes_prelaunch'], 'Insufficient disk space for the declared launch guard.')
    with np.load(P.inside(root, config['input_file']), allow_pickle=False) as inputs:
        contexts = inputs['train_contexts'].astype(np.int64).tolist()
        quality_examples = inputs['quality_sequences'].astype(np.int64)
    manifest = read(P.inside(root, config['input_manifest']))
    P.require(manifest['data_sha256'] == E.sha(P.inside(root, config['input_file'])), 'Input manifest does not bind the actual arrays.')
    P.require(manifest['model_revision'] == config['model_revision'] and manifest['corpus_revision'] == config['corpus_revision'], 'Prepared source revisions differ from the run.')
    P.require(len(contexts) == config['contexts'] and all(len(c) == config['prompt_tokens'] for c in contexts), 'Prepared prompt grid differs from the run.')
    P.require(quality_examples.shape == (manifest['quality_examples'], config['prompt_tokens'] + config['continuation_tokens']), 'Held-out target shape changed.')
    tokenizer, model = E.load_quantized(config)
    P.require(model.config.vocab_size == config['vocabulary'], 'Model vocabulary differs from the protocol.')
    base = E.distributions(model, contexts)
    with np.load(P.inside(root, config['base_conditionals']), allow_pickle=False) as frozen:
        expected = frozen['log_probabilities']
        P.require(expected.shape == base[1].shape, 'Frozen base shape changed.')
        replay_error = float(np.max(np.abs(base[1] - expected)))
    P.require(replay_error <= config['base_replay_tolerance'], 'PINNED_BASE_LOG_PROBABILITY_REPLAY_FAILED')
    parameters, modules = E.install_lora(model, config['rank'], config['alpha'])
    P.require(len(modules) == config['adapter_modules'] and sum(p.numel() for p in parameters) == config['trainable_parameters'], 'Q/V adapter grid differs from the protocol.')
    base_quality = M.heldout_quality(model, quality_examples, config['prompt_tokens'])
    runtime = dict(started_utc=E.utc(), pid=os.getpid(), HEAD_at_start=verified['committed_HEAD'],
                   precommitment_sha256=verified['precommitment_file_sha256'],
                   base_log_probability_replay_error=replay_error, baseline_quality=base_quality,
                   CUDA_device=torch.cuda.get_device_name(0), lora_modules=modules,
                   trainable_parameters=sum(p.numel() for p in parameters),
                   expected_package_versions=config['expected_package_versions'],
                   scientific_verdict_authorized=False)
    runtime_path = output / 'runtime.json'
    if runtime_path.exists():
        old = read(runtime_path)
        P.require(old['precommitment_sha256'] == runtime['precommitment_sha256'], 'Output directory belongs to a different experiment.')
        runtime['previous_runtime'] = old
    E.save_json(runtime_path, runtime)
    for job in record['design']['jobs']:
        P.verify(root, args.precommitment, require_committed=True)
        if not run_job(model, parameters, contexts, quality_examples, base, base_quality,
                       job, config, runtime['precommitment_sha256'], output):
            E.log(dict(status='RESOURCE_PAUSE', scientific_verdict_authorized=False))
            return
    E.save_json(output / 'execution_complete.json', dict(status='EXECUTION_COMPLETE_AWAITS_INDEPENDENT_ANALYSIS',
                                                        completed_utc=E.utc(),
                                                        precommitment_sha256=runtime['precommitment_sha256'],
                                                        scientific_verdict_authorized=False))


if __name__ == '__main__':
    main()
