"""
AQIDA_CONSTITUTIONAL_SCOPE: gate
THEORY_BINDING_MAP:
- CORE_LOGIC: declared observations and claim boundaries remain distinct.
- elm_eulerian: finite entropy paths compose by adjacent differences.
- n_s_u: exact source bytes, independent seeds and prior commitments are evidence.
- AQiDA_Physics: finite entropy feasibility is a diagnostic, not a physical law.
CLAIM_BOUNDARY: portable input/forecast commitment and integrity checks only.
No statistical coverage, correct prediction or real-transformer rate is certified.
FALSIFICATION_TESTS: reject infeasible prefixes, seed reuse within a condition,
source mutation, path escape, uncommitted inputs and a changed commitment body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    require(isinstance(name, str) and name and '\\' not in name, 'Use relative POSIX paths.')
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Path escapes the declared root.')
    destination = (root / path).resolve()
    require(destination.is_relative_to(root.resolve()), 'Symlink escapes the declared root.')
    return destination


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_design(design):
    canonical(design)  # Reject nonfinite JSON anywhere, including optional metadata.
    require(design['format_version'] == 1, 'Unsupported design format.')
    config = design['config']
    for field in ['generations', 'vocabulary', 'prompt_tokens', 'continuation_tokens']:
        require(positive_integer(config[field]), 'Positive integer required: ' + field)
    require(config['vocabulary'] > 1, 'Vocabulary must contain at least two atoms.')
    require(config['prompt_label_policy'] in {'continuation_only', 'full_sequence'}, 'Explicit prompt-label policy required.')
    for field in ['model_revision', 'corpus_revision']:
        value = config[field]
        require(isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value), 'Exact revision required: ' + field)
    horizon = config['generations']
    jobs = design['jobs']
    require(bool(jobs), 'At least one trajectory is required.')
    require(len({j['name'] for j in jobs}) == len(jobs), 'Duplicate job name.')
    groups = {}
    for job in jobs:
        require(job['name'] and all(c.isalnum() or c in '_-' for c in job['name']), 'Unsafe job name.')
        require(isinstance(job['train'], bool), 'Explicit training/control flag required.')
        require(isinstance(job['beta'], (float, int)) and not isinstance(job['beta'], bool) and job['beta'] > 0, 'Positive finite beta required.')
        require(positive_integer(job['seed']), 'Positive seed identifier required.')
        require(len(job['sampling_seeds']) == len(job['training_seeds']) == horizon, 'Seed grid must cover the complete horizon.')
        used = [job['initialization_seed']] + job['sampling_seeds'] + job['training_seeds']
        require(all(positive_integer(seed) for seed in used), 'Positive RNG seeds required.')
        require(len(set(used)) == len(used), 'RNG seeds overlap within a trajectory.')
        condition = job['condition']
        group = groups.setdefault(condition, {'seeds': set(), 'rng': set(), 'beta': job['beta'], 'count': 0})
        require(job['seed'] not in group['seeds'] and not group['rng'].intersection(used), 'Seeds overlap within condition: ' + condition)
        require(job['beta'] == group['beta'], 'A condition contains different temperatures.')
        group['seeds'].add(job['seed'])
        group['rng'].update(used)
        group['count'] += 1
    shared = any(groups[a]['rng'].intersection(groups[b]['rng'])
                 for a in groups for b in groups if a < b)
    require(not shared or design.get('cross_condition_seed_coupling') == 'declared_paired', 'Cross-condition seed reuse needs an explicit paired design.')
    require(design.get('uncertainty_unit') == 'independent_seed_trajectory', 'Uncertainty must be computed between seed trajectories.')
    require(design.get('stopping') == 'complete_fixed_horizon', 'A fixed complete horizon is required.')
    require(bool(design['interpretation_rules']) and bool(design['quality_metric']), 'Predeclare interpretation and quality measurement.')
    forecasts = design['forecasts']
    require(len({f['condition'] for f in forecasts}) == len(forecasts), 'Duplicate condition forecast.')
    require({f['condition'] for f in forecasts} == set(groups), 'Every physical condition needs a forecast.')
    ceiling = math.log(config['vocabulary'])
    total_values = 0
    for forecast in forecasts:
        group = groups[forecast['condition']]
        require(forecast['beta'] == group['beta'] and forecast['seeds'] == group['count'], 'Forecast/job grid mismatch.')
        paths = forecast['paths']
        require({'central', 'decision_lower', 'decision_upper'} <= set(paths), 'Complete central and decision paths required.')
        for name, path in paths.items():
            H = path['H']
            require(len(H) == horizon + 1, 'Entropy path must include generation zero and every measured generation.')
            require(all(isinstance(h, (float, int)) and not isinstance(h, bool) and 0 <= h <= ceiling for h in H), 'Entropy path violates its finite vocabulary bounds.')
            D = [a - b for a, b in zip(H, H[1:])]
            if 'D_H' in path:
                require(len(path['D_H']) == horizon and all(abs(a - b) <= 1e-12 for a, b in zip(D, path['D_H'])), 'Drift path must equal adjacent entropy differences.')
            for prefix in range(1, horizon + 1):
                require(abs(math.fsum(D[:prefix]) - (H[0] - H[prefix])) <= 1e-11, 'Entropy telescope failed.')
            total_values += len(H)
        for lower, middle, upper in zip(paths['decision_lower']['H'], paths['central']['H'], paths['decision_upper']['H']):
            require(lower <= middle <= upper, 'Decision entropy envelope is unordered.')
    return dict(physical_trajectories=len(jobs), required_generations=len(jobs) * horizon,
                conditions=len(groups), entropy_values=total_values,
                cross_condition_seed_reuse=shared,
                scope='Arithmetic feasibility and declared grid only; no prediction accuracy or interval coverage certificate.')


def committed_files(root, names):
    """Exact committed source bytes: callers must use a -text Git attribute."""
    top = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for name in names:
        path = inside(root, name)
        relative = path.relative_to(top).as_posix()
        committed = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=root)
        require(committed == path.read_bytes(), 'File bytes differ from the committed version: ' + name)
    return head


def freeze(root, design, names, output):
    root = root.resolve()
    report = validate_design(design)
    require(bool(names) and len(set(names)) == len(names), 'An explicit unique source-file list is required.')
    require(not output.exists(), 'A precommitment cannot be overwritten.')
    sources = {}
    for name in sorted(names):
        file = inside(root, name)
        require(file.is_file() and file != output.resolve(), 'Invalid frozen source: ' + name)
        sources[name] = dict(sha256=sha(file), bytes=file.stat().st_size)
    record = dict(format='temperature-drift-precommitment-v1',
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  design=design, source_files=sources, prearm_arithmetic=report,
                  boundary='File-time commitment. Require a committed/archived copy before any arm; filesystem timestamps alone do not prove prior publication.')
    record['payload_sha256'] = hashlib.sha256(canonical(record)).hexdigest()
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return record


def verify(root, commitment, require_committed=False):
    record = json.loads(commitment.read_text(encoding='utf-8'))
    require(record['format'] == 'temperature-drift-precommitment-v1', 'Unknown precommitment format.')
    stored = record['payload_sha256']
    body = {k: v for k, v in record.items() if k != 'payload_sha256'}
    require(hashlib.sha256(canonical(body)).hexdigest() == stored, 'Precommitment body changed.')
    report = validate_design(record['design'])
    require(report == record['prearm_arithmetic'], 'Arithmetic report changed.')
    for name, expected in record['source_files'].items():
        path = inside(root, name)
        require(path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Frozen file changed: ' + name)
    head = None
    if require_committed:
        head = committed_files(root, list(record['source_files']) + [commitment.resolve().relative_to(root.resolve()).as_posix()])
    return dict(status='PASS_PRECOMMITMENT_INTEGRITY', payload_sha256=stored,
                precommitment_file_sha256=sha(commitment), committed_HEAD=head,
                arithmetic=report, scientific_verdict_authorized=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'freeze', 'verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--design', type=Path)
    parser.add_argument('--files', type=Path)
    parser.add_argument('--commitment', type=Path)
    parser.add_argument('--require-committed', action='store_true')
    args = parser.parse_args()
    if args.action == 'verify':
        result = verify(args.root, args.commitment, args.require_committed)
    else:
        design = json.loads(args.design.read_text(encoding='utf-8'))
        if args.action == 'validate':
            result = validate_design(design)
        else:
            names = json.loads(args.files.read_text(encoding='utf-8'))
            result = freeze(args.root, design, names, args.commitment)
            result = dict(status='FROZEN_REQUIRES_PRIOR_COMMIT_OR_ARCHIVE', payload_sha256=result['payload_sha256'],
                          file_sha256=sha(args.commitment), arithmetic=result['prearm_arithmetic'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
