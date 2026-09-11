"""
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP:
- CORE_LOGIC: artificial test values do not establish model-scale conclusions.
- elm_eulerian: test causal loss, generation composition and resume behavior.
- n_s_u: independent direct losses and exact source corruption are falsifiers.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Portable artificial CPU
software test; no external corpus, model download, calibration or CUDA allocation.
FALSIFICATION_TESTS: causal target counts, direct NLL, fixed entropy paths,
source corruption, resumed/uninterrupted equality and no-training invariance.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from . import engine as E, metrics as M, precommitment as P, run as R
from . import transfer_forecast as T, analyze as A


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('embedding', torch.arange(77, dtype=torch.float64).reshape(11, 7) / 77 - .5)
        self.register_buffer('head', torch.arange(77, dtype=torch.float64).reshape(7, 11) / 66 - .5)
        self.A = torch.nn.Parameter(torch.linspace(-.2, .2, 14, dtype=torch.float64).reshape(2, 7))
        self.B = torch.nn.Parameter(torch.zeros(7, 2, dtype=torch.float64))
        self.config = SimpleNamespace(use_cache=False, vocab_size=11)

    def enable_input_require_grads(self):
        pass

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, ids, labels=None, **kwargs):
        x = self.embedding[ids]
        h = torch.tanh(x + torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B))
        logits = h @ self.head
        loss = None if labels is None else torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Self-test output is immutable; choose a new filename.')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = []

    def check(name, okay):
        checks.append(dict(name=name, passed=bool(okay)))

    def rejects(name, fn, exception=ValueError):
        try:
            fn()
        except exception:
            check(name, True)
        else:
            check(name, False)

    ids = torch.arange(2 * 69).reshape(2, 69) % 11
    model = TinyCausal()
    logits = model(ids).logits
    labels = M.make_labels(ids, 64)
    direct = torch.nn.functional.cross_entropy(logits[:, 63:-1].reshape(-1, 11), ids[:, 64:].reshape(-1))
    masked = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1))
    check('64_token_prompt_target_count', M.count_labels(labels) == 10)
    check('direct_continuation_loss', abs(float((direct - masked).detach())) < 1e-14)
    quality = M.aggregate_log_loss(M.continuation_log_loss(logits, ids, 64))
    check('quality_equals_direct_NLL', abs(quality['nll_nats_per_token'] - float(direct.detach())) < 1e-14)
    check('perplexity_definition', quality['perplexity'] == math.exp(quality['nll_nats_per_token']))
    distinct = M.continuation_distinct_n([[99, 1, 2, 1], [88, 1, 2, 2]], 1)
    check('distinct_excludes_prompts_and_sequence_boundaries', distinct['distinct_n'] == .75 and distinct['total_ngrams'] == 4)
    contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    quality_examples = np.arange(32, dtype=np.int64).reshape(4, 8) % 11
    config = dict(generations=2, vocabulary=11, contexts=3, generated_sequences=6,
                  generation_batch=2, production_steps=7, train_batch=1, learning_rate=1e-3,
                  prompt_tokens=3, continuation_tokens=5, prompt_label_policy='continuation_only',
                  minimum_free_bytes_generation=1, checkpoint_replay_tolerance=1e-12,
                  model_revision='0' * 40, corpus_revision='1' * 40)
    job = dict(name='artificial_cpu', condition='fixture', seed=1, beta=1.05, train=True,
               initialization_seed=570001, sampling_seeds=[570101, 570102], training_seeds=[570201, 570202])
    paths = {name: dict(H=[1.2, 1.1 + offset, 1. + offset])
             for name, offset in [('central', 0), ('decision_lower', -.1), ('decision_upper', .1)]}
    design = dict(format_version=1, experiment='ARTIFICIAL_CPU_FIXTURE', config=config,
                  jobs=[job], uncertainty_unit='independent_seed_trajectory',
                  stopping='complete_fixed_horizon', interpretation_rules=dict(fixture_only=True),
                  quality_metric=dict(name='fixed_window_NLL'), forecasts=[dict(condition='fixture', beta=1.05, seeds=1, paths=paths)])
    check('finite_forecast_paths', P.validate_design(design)['entropy_values'] == 9)
    broken = copy.deepcopy(design)
    broken['forecasts'][0]['paths']['central']['H'][2] = -.1
    rejects('reject_negative_entropy', lambda: P.validate_design(broken))
    broken_seed = copy.deepcopy(design)
    broken_seed['jobs'][0]['sampling_seeds'][1] = broken_seed['jobs'][0]['sampling_seeds'][0]
    rejects('reject_repeated_RNG_seed', lambda: P.validate_design(broken_seed))

    reference = dict(source_vocabulary=101, curves=dict(artificial=dict(beta=.95, paths=dict(
        central=[1., 1.1, 1.2], decision_lower=[1., .9, 1.], decision_upper=[1., 1.3, 1.4]))))
    transferred = T.transfer_paths(reference, 17, .8, 2)['artificial']['paths']
    check('transfer_initial_anchor', all(p['H'][0] == .8 for p in transferred.values()))
    errors = []
    for name, original in reference['curves']['artificial']['paths'].items():
        for h, actual in zip(original, transferred[name]['H']):
            odds = h / (math.log(101) - h) * (.8 / (math.log(17) - .8)) / (1. / (math.log(101) - 1.))
            errors.append(abs(actual - math.log(17) * odds / (1 + odds)))
    check('transfer_independent_odds_formula', max(errors) < 1e-14)
    check('transfer_prefix_budget', all(abs(math.fsum(p['D_H'][:g]) - .8 + p['H'][g]) < 1e-14
                                       for p in transferred.values() for g in [1, 2]))
    rejects('reject_transfer_entropy_ceiling', lambda: T.transfer_paths(reference, 17, math.log(17), 2))
    independent = A.seed_interval([1., 2., 3., 4., 5.])
    check('analysis_seed_mean', independent['mean'] == 3. and independent['n'] == 5)
    check('analysis_between_seed_SE', abs(independent['standard_error'] - math.sqrt(.5)) < 1e-15)
    rejects('analysis_rejects_25_dependent_generations', lambda: A.seed_interval([.01] * 25))
    rejects('analysis_rejects_nonfinite_seed', lambda: A.seed_interval([0., 1., 2., 3., float('nan')]))
    check('analysis_retains_perplexity_overflow', A.exp_or_none(1000.) is None)

    with tempfile.TemporaryDirectory(prefix='temperature_drift_cpu_', dir=output.parent) as temporary:
        root = Path(temporary).resolve()
        if root.parent != output.parent or not root.name.startswith('temperature_drift_cpu_'):
            raise RuntimeError('Temporary cleanup target is outside the selected output folder.')
        rejects('analysis_no_partial_grid_verdict', lambda: A.complete_inventory(design, root / 'absent'), A.IncompleteRun)
        audit = A.EvidenceAudit(root)
        baseP, baseLP = E.distributions(model, contexts, device='cpu')
        independent_H = audit.probabilities(baseLP, config, 'CPU fixture')
        check('analysis_independent_entropy_reconstruction', np.max(np.abs(independent_H - E.entropy(baseP, baseLP))) < 1e-14)
        rejects('analysis_rejects_bad_probability_mass', lambda: audit.probabilities(baseLP - .1, config, 'corrupt fixture'))
        heldout = M.heldout_quality(model, quality_examples, 3, device='cpu')
        independent_nll, _ = audit.quality(heldout, 4, 5, 'CPU fixture')
        check('analysis_independent_NLL_aggregation', abs(independent_nll - heldout['nll_nats_per_token']) < 1e-14)
        changed_quality = copy.deepcopy(heldout)
        changed_quality['per_example'][0]['target_tokens'] = 4
        rejects('analysis_rejects_wrong_heldout_token_count', lambda: audit.quality(changed_quality, 4, 5, 'corrupt fixture'))
        (root / 'fixture_source.txt').write_text('Artificial CPU source only', encoding='utf-8')
        commitment = root / 'precommitment.json'
        P.freeze(root, design, ['fixture_source.txt'], commitment)
        check('freeze_and_verify', P.verify(root, commitment)['status'] == 'PASS_PRECOMMITMENT_INTEGRITY')
        (root / 'fixture_source.txt').write_text('Changed source', encoding='utf-8')
        rejects('reject_changed_frozen_source', lambda: P.verify(root, commitment))
        cached = root / 'artificial_weights'
        cached.write_bytes(b'fixture-only')
        pinned = dict(model='artificial', model_revision='a' * 40,
                      snapshot_sha256={cached.name: dict(sha256=E.sha(cached), bytes=cached.stat().st_size)})
        resolutions = []

        def resolver(model_name, filename, **kwargs):
            resolutions.append((model_name, filename, kwargs))
            return root / filename

        E.verify_snapshot_files(pinned, resolver=resolver)
        check('snapshot_lookup_pins_revision_and_offline', resolutions == [('artificial', cached.name, dict(revision='a' * 40, local_files_only=True))])
        cached.write_bytes(b'fixture-onlX')
        rejects('reject_changed_cached_weights', lambda: E.verify_snapshot_files(pinned, resolver=resolver), RuntimeError)

        def run(destination, selected_job=job):
            destination.mkdir(parents=True, exist_ok=True)
            tiny = TinyCausal()
            base = E.distributions(tiny, contexts, device='cpu')
            baseline = M.heldout_quality(tiny, quality_examples, 3, device='cpu')
            return R.run_job(tiny, list(tiny.parameters()), contexts, quality_examples, base, baseline,
                             selected_job, config, 'a' * 64, destination, device='cpu', resource_check=lambda: [])

        with redirect_stdout(io.StringIO()):
            check('uninterrupted_complete', run(root / 'full'))
            resumed = root / 'resumed'
            resumed.mkdir()
            marker = resumed / 'PAUSE_AFTER_GENERATION'
            marker.write_text('CPU fixture pause', encoding='utf-8')
            check('resource_pause', run(resumed) is False)
            marker.unlink()
            check('resume_complete', run(resumed))
            control_job = dict(job, name='artificial_control', train=False)
            check('control_complete', run(root / 'control', control_job))
        full = R.read(root / 'full' / job['name'] / 'trajectory.json')
        continued = R.read(root / 'resumed' / job['name'] / 'trajectory.json')
        control = R.read(root / 'control' / control_job['name'] / 'trajectory.json')
        for index, (a, b) in enumerate(zip(full['generations'], continued['generations']), 1):
            check(f'exact_resumed_measurements_g{index}', all(a[k] == b[k] for k in ['H_source', 'H_production', 'D_H', 'quality', 'tokens_sha256', 'conditionals_sha256']))
            check(f'optimizer_and_quality_counts_g{index}', a['training_target_labels'] == 35 and a['quality']['target_tokens'] == 20)
        check('checkpoint_log_probability_replay', continued['resumes'][0]['log_probability_max_error'] == 0.)
        check('control_entropy_and_quality_invariant', all(g['mean_D_H'] == 0. and g['quality_NLL_change_from_base'] == 0. for g in control['generations']))
        check('no_scientific_verdict', not continued['scientific_verdict_authorized'])
    check('temporary_cleanup_confined', not root.exists())
    check('no_CUDA_initialized', not torch.cuda.is_initialized())
    result = dict(status='PASS_PORTABLE_CPU_SELF_TEST' if all(c['passed'] for c in checks) else 'FAIL',
                  passed=sum(c['passed'] for c in checks), total=len(checks), checks=checks,
                  finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['self_test.py', 'engine.py', 'metrics.py', 'precommitment.py', 'run.py', 'transfer_forecast.py', 'analyze.py']},
                  CUDA_initialized=torch.cuda.is_initialized(), no_external_data_or_model=True,
                  fixture_only=True, scientific_verdict_authorized=False)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'source_sha256']}, indent=2))
    if result['status'] != 'PASS_PORTABLE_CPU_SELF_TEST':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
