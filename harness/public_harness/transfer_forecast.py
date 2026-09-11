"""
AQIDA_CONSTITUTIONAL_SCOPE: diagnostic
THEORY_BINDING_MAP:
- CORE_LOGIC: normalized entropy retains its comparator measurement boundary.
- elm_eulerian: transfer whole paths and derive every adjacent drift exactly.
- n_s_u: preserve an old frozen response hypothesis instead of fitting new outcomes.
- e_i/AQiDA_Physics: expose cross-setting forecast residuals as diagnostics.
CLAIM_BOUNDARY: bounded empirical forecast transfer only; no universal temperature
law, new-family confidence coverage or scalar phase-truth interpretation.
FALSIFICATION_TESTS: original-grid identity, target initial anchor, finite prefix
bounds, envelope ordering, independent odds-ratio computation and no clipping.
"""
from __future__ import annotations

import math


def normalized_logit(entropy, ceiling):
    fraction = entropy / ceiling
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('A representable interior normalized entropy is required.')
    return math.log(fraction) - math.log1p(-fraction)


def entropy_from_logit(coordinate, ceiling):
    if coordinate >= 0:
        return ceiling / (1 + math.exp(-coordinate))
    exponential = math.exp(coordinate)
    return ceiling * exponential / (1 + exponential)


def transfer_paths(reference, target_vocabulary, target_initial_entropy, horizon):
    if not isinstance(target_vocabulary, int) or target_vocabulary <= 1:
        raise ValueError('An integer vocabulary greater than one is required.')
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('A positive integer horizon is required.')
    old_ceiling = math.log(reference['source_vocabulary'])
    new_ceiling = math.log(target_vocabulary)
    target_coordinate = normalized_logit(target_initial_entropy, new_ceiling)
    output = {}
    for condition, row in reference['curves'].items():
        initial = row['paths']['central'][0]
        shift = target_coordinate - normalized_logit(initial, old_ceiling)
        paths = {}
        for name in ['central', 'decision_lower', 'decision_upper']:
            old = row['paths'][name]
            if len(old) < horizon + 1 or old[0] != initial:
                raise ValueError('Reference paths need the same initial anchor and the requested horizon.')
            transferred = [entropy_from_logit(normalized_logit(h, old_ceiling) + shift, new_ceiling)
                           for h in old[:horizon + 1]]
            # This is the supplied initial condition, not a fitted or clipped
            # outcome. Mathematically the coordinate map fixes it exactly.
            if abs(transferred[0] - target_initial_entropy) > 1e-12:
                raise ValueError('Initial coordinate transfer is numerically unstable.')
            transferred[0] = target_initial_entropy
            if not all(math.isfinite(h) and 0 <= h <= new_ceiling for h in transferred):
                raise ValueError('Transferred path violates finite entropy bounds.')
            drift = [a - b for a, b in zip(transferred, transferred[1:])]
            if any(abs(math.fsum(drift[:n]) - target_initial_entropy + transferred[n]) > 1e-11
                   for n in range(1, horizon + 1)):
                raise ValueError('Transferred entropy telescope failed.')
            paths[name] = dict(H=transferred, D_H=drift)
        if not all(lo <= mid <= hi for lo, mid, hi in
                   zip(paths['decision_lower']['H'], paths['central']['H'], paths['decision_upper']['H'])):
            raise ValueError('Transferred decision envelope is unordered.')
        output[condition] = dict(beta=row['beta'], paths=paths,
                                 method='Common translation in normalized-entropy logit coordinates',
                                 coordinate_shift=shift,
                                 boundary='Transferred frozen empirical paths; no new-family fit or coverage guarantee.')
    return output
