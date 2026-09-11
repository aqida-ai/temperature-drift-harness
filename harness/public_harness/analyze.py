"""
AQIDA_CONSTITUTIONAL_SCOPE: gate
THEORY_BINDING_MAP:
- CORE_LOGIC: reconstruct normalized comparator observations and signed drifts.
- elm_eulerian: preserve complete teacher/sample/reset/checkpoint composition.
- n_s_u: complete independent seed trajectories are the uncertainty units.
- AQiDA_Physics/Assessment for Genenral AI: held-out objective evidence stays separate.
CLAIM_BOUNDARY: finite five-generation, five-seed near-pair and three-control
analysis. No universal temperature, actual-collapse, quality-equivalence, Fisher,
rate, public-priority or phase-truth claim. Saved NLL arithmetic is checked;
the model's held-out forward passes are not independently recomputed here.
FALSIFICATION_TESTS: partial grids, source mutation, incorrect teacher chains,
invalid probabilities, incorrect target counts, pseudoreplication and discordant quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from . import precommitment as P

T95 = {3: 4.302652729696142, 5: 2.7764451051977987}
CONDITIONS = {'near_warm': 5, 'near_cool': 5, 'no_training': 3}


class IncompleteRun(ValueError):
    pass


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            P.require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)

    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exp_or_none(value):
    try:
        return math.exp(value)
    except OverflowError:
        return None


def seed_interval(values):
    values = list(values)
    n = len(values)
    P.require(n in T95 and all(finite(x) for x in values), 'Expected three or five finite independent seed values.')
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    half = T95[n] * se
    return dict(n=n, df=n - 1, values=values, mean=mean, standard_error=se,
                critical_value=T95[n], lower=mean - half, upper=mean + half,
                uncertainty_unit='independent_complete_seed_trajectory')


def summarize(design, rows, base_entropy, baseline_nll):
    """Pure fixed-profile analysis; callers must first audit all raw evidence."""
    P.validate_design(design)
    P.require(design['config']['generations'] == 5, 'Analysis profile requires the frozen five-generation horizon.')
    P.require(design['interpretation_rules']['t95_df4'] == T95[5], 'Primary critical value changed.')
    P.require(set(j['condition'] for j in design['jobs']) == set(CONDITIONS), 'Analysis condition profile changed.')
    P.require(all(j['train'] is (j['condition'] != 'no_training') for j in design['jobs']),
              'Analysis profile training/control assignment changed.')
    expected_names = {j['name'] for j in design['jobs']}
    P.require(set(rows) == expected_names, 'Missing, duplicate or extra complete trajectory.')
    result = {}
    horizon = 5
    forecasts = {f['condition']: f for f in design['forecasts']}
    for condition, count in CONDITIONS.items():
        jobs = sorted((j for j in design['jobs'] if j['condition'] == condition), key=lambda j: j['seed'])
        P.require(len(jobs) == count and len({j['seed'] for j in jobs}) == count,
                  'Independent seed count changed: ' + condition)
        selected = [rows[j['name']] for j in jobs]
        for item in selected:
            P.require(len(item['generations']) == horizon, 'Partial trajectory cannot enter analysis.')
            P.require([g['generation'] for g in item['generations']] == list(range(1, horizon + 1)),
                      'Generation grid is not complete and ordered.')
        drift = seed_interval(statistics.fmean(g['D_H'] for g in item['generations']) for item in selected)
        H = [base_entropy] + [statistics.fmean(item['generations'][i]['H'] for item in selected)
                              for i in range(horizon)]
        for item in selected:
            P.require(abs(math.fsum(g['D_H'] for g in item['generations']) -
                          (base_entropy - item['generations'][-1]['H'])) <= 1e-10,
                      'Seed drift does not reconstruct its terminal entropy.')
        central = forecasts[condition]['paths']['central']['H']
        lower = forecasts[condition]['paths']['decision_lower']['H']
        upper = forecasts[condition]['paths']['decision_upper']['H']
        inside = [lo <= h <= hi for lo, h, hi in zip(lower, H, upper)]
        # Generation zero is shared initialization, not an additional prediction.
        curves = dict(H_seed_mean=H, central=central, decision_lower=lower,
                      decision_upper=upper, inside_by_generation=inside,
                      covered_points_including_initial=sum(inside), total_points_including_initial=6,
                      covered_postinitial_points=sum(inside[1:]), total_postinitial_points=5,
                      complete_postinitial_agreement=all(inside[1:]),
                      max_absolute_central_error=max(abs(a - b) for a, b in zip(H, central)),
                      coverage_guarantee=False)
        quality = []
        per_generation = []
        for i in range(horizon):
            per_generation.append(dict(generation=i + 1,
                D_H=seed_interval(item['generations'][i]['D_H'] for item in selected),
                H=seed_interval(item['generations'][i]['H'] for item in selected)))
            nll = seed_interval(item['generations'][i]['NLL'] for item in selected)
            delta = seed_interval(item['generations'][i]['NLL'] - baseline_nll for item in selected)
            outcome = ('NLL_INCREASE_RESOLVED' if delta['lower'] > 0 else
                       'NLL_DECREASE_RESOLVED' if delta['upper'] < 0 else 'NLL_CHANGE_UNRESOLVED')
            quality.append(dict(generation=i + 1, NLL=nll, NLL_change_from_base=delta,
                perplexity_of_token_weighted_seed_mean_NLL=exp_or_none(nll['mean']),
                NLL_direction=outcome,
                distinct_2=seed_interval(item['generations'][i]['distinct_2'] for item in selected)))
        result[condition] = dict(seed_ids=[j['seed'] for j in jobs], trajectory_mean_D_H=drift,
                                 per_generation=per_generation, curves=curves, quality=quality,
                                 terminal_quality=quality[-1])
    warm = result['near_warm']['trajectory_mean_D_H']
    cool = result['near_cool']['trajectory_mean_D_H']
    controls = [rows[j['name']] for j in design['jobs'] if j['condition'] == 'no_training']
    tolerance = design['interpretation_rules']['control_replay_tolerance']
    max_control_H = max(abs(g['H'] - base_entropy) for r in controls for g in r['generations'])
    max_control_NLL = max(abs(g['NLL'] - baseline_nll) for r in controls for g in r['generations'])
    max_control_context_H = max(r['max_control_context_entropy_error'] for r in controls)
    max_control_quality = max(r['max_control_per_example_NLL_error'] for r in controls)
    control = max(max_control_H, max_control_NLL, max_control_context_H, max_control_quality) <= tolerance
    direction = warm['upper'] < 0 and cool['lower'] > 0
    curves = all(result[c]['curves']['complete_postinitial_agreement'] for c in ['near_warm', 'near_cool'])
    if not control:
        verdict = 'FIXED_MODEL_CONTROL_FAILED'
    elif not direction:
        verdict = 'DIRECTIONAL_TRANSFER_UNRESOLVED_AT_FIXED_HORIZON'
    elif not curves:
        verdict = 'DIRECTIONAL_TRANSFER_RESOLVED_CURVES_MISSED'
    else:
        verdict = 'DIRECTIONAL_AND_COMPLETE_CURVE_TRANSFER_OBSERVED'
    return dict(verdict=verdict, conditions=result,
        primary=dict(warmer_upper_below_zero=warm['upper'] < 0, cooler_lower_above_zero=cool['lower'] > 0,
                     both_directional_intervals=direction),
        complete_curve_transfer=curves,
        control=dict(passed=control, tolerance=tolerance, max_entropy_change=max_control_H,
                     max_NLL_change=max_control_NLL, max_context_entropy_change=max_control_context_H,
                     max_per_example_NLL_change=max_control_quality,
                     distinct_2_invariance_required=False),
        quality_separate_from_entropy=True, quality_equivalence_authorized=False,
        actual_collapse_claim_authorized=False, exact_new_family_critical_beta_authorized=False,
        isolated_architecture_causality_authorized=False, public_claim_authorized=False,
        scope='Finite combined model-family, context-length and prompt-label-policy transfer; no new-family interval-coverage guarantee.')


class EvidenceAudit:
    def __init__(self, run):
        self.run = run.resolve()
        self.checks = 0
        self.files = {}
        self.max_numeric_error = 0.

    def check(self, condition, message):
        P.require(condition, message)
        self.checks += 1

    def close(self, actual, expected, message, atol=1e-10, rtol=0.):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message + ': shape/finite')
        error = float(np.max(np.abs(a - b))) if a.size else 0.
        self.max_numeric_error = max(self.max_numeric_error, error)
        self.check(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)), message + ': value')

    def bind(self, path, expected=None):
        path = path.resolve()
        self.check(path.is_relative_to(self.run) and path.is_file(), 'Evidence file missing or escapes run: ' + str(path))
        digest = P.sha(path)
        if expected is not None:
            self.check(digest == expected, 'Evidence SHA256 changed: ' + str(path))
        self.files[path.relative_to(self.run).as_posix()] = dict(sha256=digest, bytes=path.stat().st_size)
        return digest

    def json(self, path, expected=None):
        self.bind(path, expected)
        return read(path)

    def probabilities(self, lp, config, label):
        self.check(lp.dtype == np.float64 and lp.shape == (config['contexts'], config['vocabulary']), label + ': conditional grid')
        self.check(np.isfinite(lp).all() and np.all(lp <= 0), label + ': finite log probabilities')
        mass = np.exp(lp)
        self.close(np.sum(mass, axis=1), np.ones(config['contexts']), label + ': normalized', atol=1e-12)
        # Independent of the production engine; summation uses math.fsum per context.
        H = np.array([-math.fsum(float(p * l) for p, l in zip(pr, lr)) for pr, lr in zip(mass, lp)])
        self.check(np.all(H >= 0) and np.all(H <= math.log(config['vocabulary']) + 1e-12), label + ': entropy bounds')
        return H

    def quality(self, value, examples, continuation, label):
        rows = value['per_example']
        self.check(len(rows) == examples and [r['example'] for r in rows] == list(range(examples)), label + ': example grid')
        self.check(all(type(r['target_tokens']) is int and r['target_tokens'] == continuation and
                       finite(r['nll_sum']) and r['nll_sum'] >= 0 for r in rows), label + ': labels and losses')
        total = examples * continuation
        loss_sum = math.fsum(r['nll_sum'] for r in rows)
        nll = loss_sum / total
        self.check(value['target_tokens'] == total, label + ': token count')
        self.close(value['nll_sum'], loss_sum, label + ': summed NLL', atol=1e-10, rtol=1e-13)
        self.close(value['nll_nats_per_token'], nll, label + ': weighted NLL')
        ppl = exp_or_none(nll)
        self.check(value['perplexity_overflow'] is (ppl is None), label + ': overflow declaration')
        if ppl is None:
            self.check(value['perplexity'] is None, label + ': retain overflow')
        else:
            self.close(value['perplexity'], ppl, label + ': perplexity', rtol=1e-13)
        return nll, np.array([r['nll_sum'] / continuation for r in rows])

    def checkpoint(self, path, expected, config, initial_shapes=None, initial=False):
        import torch
        self.bind(path, expected)
        values = torch.load(path, map_location='cpu', weights_only=True)
        self.check(isinstance(values, list) and len(values) == 2 * config['adapter_modules'], 'Adapter factor count changed.')
        self.check(all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.dtype == torch.float32 and
                       v.ndim == 2 and bool(torch.isfinite(v).all()) for v in values), 'Invalid adapter tensor.')
        shapes = [list(v.shape) for v in values]
        self.check(sum(v.numel() for v in values) == config['trainable_parameters'], 'Adapter parameter count changed.')
        self.check(all(values[i].shape[0] == values[i + 1].shape[1] == config['rank']
                       for i in range(0, len(values), 2)), 'Adapter rank changed.')
        if initial_shapes is not None:
            self.check(shapes == initial_shapes, 'Adapter factor shapes changed.')
        if initial:
            self.check(all(bool((values[i] == 0).all()) for i in range(1, len(values), 2)), 'Initial B factor is not zero.')
        self.check(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA.')
        return shapes


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    P.require(parsed.tzinfo is not None, 'Evidence timestamp needs an explicit timezone.')
    return parsed


def complete_inventory(design, run):
    """Inspect completion before reading outcomes or calculating any intervals."""
    records, pending = {}, []
    expected = {j['name'] for j in design['jobs']}
    observed = {p.parent.name for p in run.glob('*/trajectory.json')}
    P.require(observed <= expected, 'Run contains an undeclared trajectory.')
    for job in design['jobs']:
        path = run / job['name'] / 'trajectory.json'
        if not path.is_file():
            pending.append(job['name'] + ': absent')
            continue
        row = read(path)
        P.require(row['job'] == job, 'Recorded job differs from the frozen design: ' + job['name'])
        P.require(len(row['generations']) <= design['config']['generations'], 'Extra generation exceeds the frozen horizon.')
        if len(row['generations']) < design['config']['generations'] or row['status'] in {'RUNNING', 'RESOURCE_PAUSE'}:
            pending.append(job['name'] + ': ' + str(len(row['generations'])))
        else:
            P.require(row['status'] == 'COMPLETE_TRAJECTORY_AWAITS_ANALYSIS', 'Unexpected completed trajectory status.')
        records[job['name']] = row
    if pending or not (run / 'execution_complete.json').is_file() or not (run / 'runtime.json').is_file():
        raise IncompleteRun('Complete grid and execution marker required before analysis. ' + '; '.join(pending))
    return records


def audit_run(root, commitment, commitment_sha, run):
    """Independent CPU audit of a complete saved run; never imports engine/metrics."""
    import torch
    torch.set_num_threads(1)
    P.require(not torch.cuda.is_initialized(), 'Audit must remain on CPU.')
    design = commitment['design']
    P.validate_design(design)
    config = design['config']
    records = complete_inventory(design, run)
    audit = EvidenceAudit(run)
    audit.check(config['prompt_label_policy'] == 'continuation_only' and config['train_batch'] == 1 and
                config['continuation_tokens'] >= 2,
                'This audit profile requires continuation-only, single-sequence training.')
    with np.load(P.inside(root, config['input_file']), allow_pickle=False) as stored:
        contexts = stored['train_contexts']
        quality_examples = stored['quality_sequences']
    audit.check((contexts.dtype == quality_examples.dtype and contexts.dtype in (np.dtype("int32"), np.dtype("int64"))) and
                contexts.shape == (config['contexts'], config['prompt_tokens']), 'Prepared context grid changed.')
    quality_count = design['quality_metric']['examples']
    audit.check(quality_examples.shape == (quality_count, config['prompt_tokens'] + config['continuation_tokens']),
                'Prepared held-out grid changed.')
    audit.check(design['quality_metric']['labels_per_evaluation'] == quality_count * config['continuation_tokens'] and
                design['quality_metric']['evaluation_beta'] == 1., 'Quality target count or evaluation temperature changed.')
    manifest = read(P.inside(root, config['input_manifest']))
    audit.check(manifest['data_sha256'] == P.sha(P.inside(root, config['input_file'])) and
                manifest['model_revision'] == config['model_revision'] and
                manifest['corpus_revision'] == config['corpus_revision'], 'Prepared input provenance changed.')
    for name, tokens in [('training prompts', contexts), ('held-out windows', quality_examples)]:
        audit.check(int(tokens.min()) >= 0 and int(tokens.max()) < config['vocabulary'], name + ': token outside vocabulary')
    with np.load(P.inside(root, config['base_conditionals']), allow_pickle=False) as stored:
        baseLP = stored['log_probabilities']
    baseH = audit.probabilities(baseLP, config, 'base')
    base_mean = statistics.fmean(baseH)
    audit.close(design['base_context_entropies'], baseH, 'Frozen context entropy')
    audit.close(design['base_initial_entropy'], base_mean, 'Frozen initial entropy')
    runtime = audit.json(run / 'runtime.json')
    baseline_nll, baseline_examples = audit.quality(runtime['baseline_quality'], quality_count,
                                                   config['continuation_tokens'], 'baseline quality')
    runtime_chain = []
    current = runtime
    while current is not None:
        audit.check(current['precommitment_sha256'] == commitment_sha and
                    current['scientific_verdict_authorized'] is False, 'Runtime commitment or claim boundary changed.')
        audit.check(current['trainable_parameters'] == config['trainable_parameters'] and
                    len(current['lora_modules']) == config['adapter_modules'] and
                    len(set(current['lora_modules'])) == config['adapter_modules'], 'Runtime adapter grid changed.')
        audit.check(current['expected_package_versions'] == config['expected_package_versions'], 'Runtime package record changed.')
        audit.check(finite(current['base_log_probability_replay_error']) and
                    0 <= current['base_log_probability_replay_error'] <= config['base_replay_tolerance'],
                    'Pinned base replay failed.')
        old_nll, old_examples = audit.quality(current['baseline_quality'], quality_count,
                                             config['continuation_tokens'], 'runtime baseline')
        audit.close(old_examples, baseline_examples, 'Resumed baseline quality', atol=config['base_replay_tolerance'])
        audit.check(timestamp(current['started_utc']) >= timestamp(commitment['created_utc']), 'Runtime predates commitment.')
        runtime_chain.append(dict(HEAD_at_start=current['HEAD_at_start'], started_utc=current['started_utc']))
        current = current.get('previous_runtime')
    complete = audit.json(run / 'execution_complete.json')
    audit.check(complete['status'] == 'EXECUTION_COMPLETE_AWAITS_INDEPENDENT_ANALYSIS' and
                complete['precommitment_sha256'] == commitment_sha and
                complete['scientific_verdict_authorized'] is False, 'Completion marker does not match the run.')
    rows = {}
    trained_count = control_count = labels = 0
    for job in design['jobs']:
        folder = run / job['name']
        record = records[job['name']]
        audit.bind(folder / 'trajectory.json')
        audit.check(record['precommitment_sha256'] == commitment_sha and
                    record['scientific_verdict_authorized'] is False, 'Trajectory commitment changed.')
        start_time = timestamp(record['started_utc'])
        audit.check(start_time >= timestamp(commitment['created_utc']), 'Trajectory predates commitment.')
        for resume in record.get('resumes', []):
            audit.check(type(resume['after_generation']) is int and 0 <= resume['after_generation'] <= config['generations'] and
                        finite(resume['log_probability_max_error']) and
                        0 <= resume['log_probability_max_error'] <= config['checkpoint_replay_tolerance'], 'Resume checkpoint replay failed.')
        initial = audit.json(folder / 'initial_adapter.json')
        audit.check(initial['seed'] == job['initialization_seed'], 'Initialization seed changed.')
        shapes = audit.checkpoint(folder / 'initial_adapter.pt', initial['sha256'], config, initial=True)
        teacher = initial['sha256']
        sourceLP, sourceH = baseLP, baseH
        cumulative = np.zeros(config['contexts'], dtype=np.float64)
        values = []
        control_context_error = control_quality_error = 0.
        prior_time = start_time
        expected_folders = {f'g{i:02d}' for i in range(1, config['generations'] + 1)}
        actual_folders = {p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith('g')}
        audit.check(actual_folders == expected_folders, 'Generation folder inventory changed.')
        for index, recorded in enumerate(record['generations'], 1):
            generation = folder / f'g{index:02d}'
            audit.check(recorded['generation'] == index and recorded['scientific_verdict_authorized'] is False,
                        'Generation order or claim boundary changed.')
            completed_time = timestamp(recorded['completed_utc'])
            audit.check(completed_time >= prior_time, 'Generation time order changed.')
            prior_time = completed_time
            stage = audit.json(generation / 'stage.json')
            audit.check(stage['status'] == 'MEASURED' and stage['generation'] == index and
                        stage['precommitment_sha256'] == commitment_sha and
                        stage['teacher_adapter_sha256'] == teacher and
                        stage['sampling_seed'] == job['sampling_seeds'][index - 1] and
                        stage['training_seed'] == job['training_seeds'][index - 1] and
                        stage['completed_utc'] == recorded['completed_utc'], 'Stage teacher/seed/status binding changed.')
            audit.check(timestamp(stage['started_utc']) >= start_time and
                        timestamp(stage['started_utc']) <= completed_time, 'Stage time order changed.')
            audit.check(stage['tokens_sha256'] == recorded['tokens_sha256'], 'Sample stage hash changed.')
            audit.bind(generation / 'tokens.npz', recorded['tokens_sha256'])
            with np.load(generation / 'tokens.npz', allow_pickle=False) as stored:
                sequences = stored['tokens']
            audit.check(sequences.dtype == np.int64 and sequences.shape ==
                        (config['generated_sequences'], config['prompt_tokens'] + config['continuation_tokens']), 'Generated token shape changed.')
            audit.check(int(sequences.min()) >= 0 and int(sequences.max()) < config['vocabulary'], 'Generated token outside vocabulary.')
            audit.check(np.array_equal(sequences[:, :config['prompt_tokens']],
                        contexts[np.arange(len(sequences)) % len(contexts)]), 'Generated prompt cycle changed.')
            pairs = {(int(a), int(b)) for row in sequences[:, config['prompt_tokens']:]
                     for a, b in zip(row, row[1:])}
            total_pairs = len(sequences) * (config['continuation_tokens'] - 1)
            diversity = len(pairs) / total_pairs
            saved_diversity = recorded['generated_distinct_2']
            audit.check(saved_diversity['n'] == 2 and saved_diversity['unique_ngrams'] == len(pairs) and
                        saved_diversity['total_ngrams'] == total_pairs, 'Continuation distinct-2 counts changed.')
            audit.close(saved_diversity['distinct_n'], diversity, 'Continuation distinct-2 ratio', atol=1e-14)
            if job['train']:
                training = audit.json(generation / 'training.json', recorded['training_sha256'])
                audit.check(training['tokens_sha256'] == recorded['tokens_sha256'] and
                            training['precommitment_sha256'] == commitment_sha and
                            training['adapter_sha256'] == recorded['adapter_sha256'], 'Training evidence binding changed.')
                audit.check(training['reset_policy'] == 'same per-trajectory initial adapter before every fit', 'Reset policy changed.')
                schedule = training['training_indices']
                generator = torch.Generator(device='cpu').manual_seed(job['training_seeds'][index - 1])
                expected_schedule = [int(torch.randint(0, len(sequences), (1,), generator=generator)[0])
                                     for _ in range(config['production_steps'])]
                audit.check(schedule == expected_schedule and all(type(i) is int for i in schedule), 'Training schedule does not replay from its seed.')
                audit.check(len(training['loss_trace']) == config['production_steps'] and
                            all(finite(v) and v >= 0 for v in training['loss_trace']), 'Invalid optimizer loss trace.')
                budget = config['production_steps'] * config['continuation_tokens']
                audit.check(training['consumed_target_labels'] == recorded['training_target_labels'] == budget and
                            training['label_budget'] == dict(continuation_labels=budget, fixed_prompt_labels=0,
                                                            total_labels=budget, prompt_label_policy='continuation_only'),
                            'Training target budget changed.')
                audit.checkpoint(generation / 'adapter.pt', recorded['adapter_sha256'], config, initial_shapes=shapes)
                teacher = recorded['adapter_sha256']
                labels += budget
                trained_count += 1
            else:
                audit.check(recorded['training_target_labels'] == 0 and recorded['training_sha256'] is None and
                            recorded['adapter_sha256'] == initial['sha256'] and
                            not (generation / 'training.json').exists() and not (generation / 'adapter.pt').exists(),
                            'No-training control contains optimizer/checkpoint evidence.')
                control_count += 1
            audit.bind(generation / 'conditionals.npz', recorded['conditionals_sha256'])
            with np.load(generation / 'conditionals.npz', allow_pickle=False) as stored:
                targetLP = stored['log_probabilities']
            targetH = audit.probabilities(targetLP, config, job['name'] + f':g{index}')
            tilted = job['beta'] * sourceLP
            maximum = np.max(tilted, axis=1, keepdims=True)
            normalizer = maximum + np.log(np.exp(tilted - maximum).sum(axis=1, keepdims=True))
            decodedH = audit.probabilities(tilted - normalizer, config, 'decoded diagnostic')
            D, J = sourceH - targetH, sourceH - decodedH
            for field, reconstructed in [('H_source', sourceH), ('H_production', targetH),
                                          ('H_decoded', decodedH), ('D_H', D), ('J_H', J), ('signed_budget', J - D)]:
                audit.close(recorded[field], reconstructed, 'Reconstructed ' + field)
            mean_D, mean_H = statistics.fmean(D), statistics.fmean(targetH)
            audit.close(recorded['mean_D_H'], mean_D, 'Mean drift')
            audit.close(recorded['mean_H'], mean_H, 'Mean entropy')
            cumulative += D
            audit.close(cumulative, baseH - targetH, 'All-prefix entropy telescope')
            nll, example_nll = audit.quality(recorded['quality'], quality_count, config['continuation_tokens'], 'generation quality')
            audit.close(recorded['quality_NLL_change_from_base'], nll - baseline_nll, 'NLL change from fixed base',
                        atol=config['base_replay_tolerance'])
            if not job['train']:
                control_context_error = max(control_context_error, float(np.max(np.abs(targetH - baseH))))
                control_quality_error = max(control_quality_error, float(np.max(np.abs(example_nll - baseline_examples))))
            values.append(dict(generation=index, H=mean_H, D_H=mean_D, NLL=nll,
                               perplexity=exp_or_none(nll), distinct_2=diversity))
            sourceLP, sourceH = targetLP, targetH
        audit.check(timestamp(record['completed_utc']) >= prior_time and
                    timestamp(complete['completed_utc']) >= timestamp(record['completed_utc']), 'Run completion time order changed.')
        rows[job['name']] = dict(job=job, generations=values,
            max_control_context_entropy_error=control_context_error,
            max_control_per_example_NLL_error=control_quality_error)
    summary = summarize(design, rows, base_mean, baseline_nll)
    # Recheck every raw evidence byte after reading, to reject an active writer.
    for name, item in audit.files.items():
        audit.check(P.sha(run / name) == item['sha256'], 'Evidence changed during analysis: ' + name)
    audit.check(not torch.cuda.is_initialized(), 'CPU analysis initialized CUDA.')
    return dict(status='PASS_MEASURED', instrument_valid=True, checks_passed=audit.checks,
                physical_trajectories=len(rows), measured_generations=trained_count + control_count,
                trained_generations=trained_count, untrained_generations=control_count,
                training_target_labels=labels, max_arithmetic_discrepancy=audit.max_numeric_error,
                baseline_entropy=base_mean, baseline_NLL=baseline_nll,
                evidence_files=audit.files, runtime_chain=runtime_chain, reconstructed=rows,
                analysis=summary, CUDA_initialized=False,
                quality_audit_scope='Saved per-example loss arithmetic only; no independent model forward reevaluation.',
                public_claim_authorized=False)


def verify_runtime_commits(root, commitment_path, commitment, chain):
    """Each launch HEAD must already contain every exact precommitment source."""
    top = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip())
    files = list(commitment['source_files']) + [commitment_path.resolve().relative_to(root).as_posix()]
    for head in sorted({row['HEAD_at_start'] for row in chain}):
        P.require(len(head) == 40 and all(c in '0123456789abcdef' for c in head), 'Runtime HEAD must be an exact commit.')
        for name in files:
            path = P.inside(root, name)
            blob = subprocess.check_output(['git', 'show', head + ':' + path.relative_to(top).as_posix()], cwd=root)
            P.require(hashlib.sha256(blob).hexdigest() == P.sha(path), 'Runtime launch HEAD did not contain frozen input: ' + name)
    return dict(launch_commits=len({row['HEAD_at_start'] for row in chain}), files_per_commit=len(files),
                scope='Exact local Git bytes at recorded launch commits; no independent public timestamp claim.')


def analyze_committed(root, commitment_path, run):
    root, commitment_path, run = root.resolve(), commitment_path.resolve(), run.resolve()
    verified = P.verify(root, commitment_path, require_committed=True)
    commitment = read(commitment_path)
    self_name = Path(__file__).resolve().relative_to(root).as_posix()
    P.require(self_name in commitment['source_files'], 'Independent analyzer was not part of the prearm commitment.')
    result = audit_run(root, commitment, verified['precommitment_file_sha256'], run)
    result['launch_commit_audit'] = verify_runtime_commits(root, commitment_path, commitment, result['runtime_chain'])
    # The captured comparison is against a frozen commit, so a peer HEAD advance
    # is harmless provided all frozen sources still verify at the end.
    result['final_precommitment_check'] = P.verify(root, commitment_path, require_committed=True)
    result['precommitment_sha256'] = verified['precommitment_file_sha256']
    result['analyzer_sha256'] = P.sha(Path(__file__))
    result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--precommitment', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    P.require(not args.output.exists(), 'Preserve prior analysis receipts; choose a new output path.')
    try:
        result = analyze_committed(args.root, args.precommitment, args.run)
        code = 0
    except IncompleteRun as error:
        result = dict(status='INCOMPLETE', reason=str(error), scientific_verdict_authorized=False)
        code = 2
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        result = dict(status='FAIL_INSTRUMENT', reason=str(error), scientific_verdict_authorized=False)
        code = 1
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'evidence_files', 'reconstructed', 'analysis'}}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
