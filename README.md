<!--
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP: CORE_LOGIC signed measurement; elm_eulerian exact composition;
n_s_u frozen source and evidence receipts; e_i residual and assessment scope.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic; no public-priority claim.
FALSIFICATION_TESTS: byte manifests, complete seed gates, adverse receipts, CPU fixtures.
-->
# Temperature-drift preregistration harness — v0.4.1

## What this is

A preregistration harness for recursive-training drift experiments on 7B models,
with the audit/freeze/receipt architecture used across Sprints 53C, 54 and 55.
It records frozen predictions, complete independent seed trajectories, signed
entropy drift and separate quality measurements. This is a baseline/comparator.

## Precommitment chain

| Record | Full identifier | Identifier type |
| --- | --- | --- |
| S53C prearm source commit | `7de2927f3916fa24c878179a881fe167ef54b182` | Local Git commit SHA-1 |
| S54 precommitment | `108e8f4c33fa14e705cc734ae950ffd09685a63929dc9db3ef7faaa8086124b1` | File SHA-256 |
| S55 precommitment | `a7bf777d07faba017420e550a66f87d671d3c30409663148e97c7103881aa997` | File SHA-256 |

The exact commitment files and their hashes are in [the evidence index](evidence/README.md).
The inherited beta_c is 0.9974153735881554 (rounded display 0.9974), with the
same 1.04 temperature-ratio protocol. These local prearm records do not establish
independently timestamped public preregistration or public priority.

## Refusal ledger

laboratory "predicts-and-prevents" REFUSED (collapse endpoint failed by deceleration; seed-2 rebound at generation 14); public-priority REFUSED; first-in-field REFUSED.

These refusals are the method working: a successful instrument audit does not
erase an adverse scientific outcome or authorize a stronger headline.

## Instrument-repair disclosure

The original S55 frozen analyzer returned **FAIL_INSTRUMENT** before aggregate
statistics: it required int64 stored tokens while the frozen preparation saved
int32. The original context and held-out arrays matched their frozen hashes.
A separately committed repair changed one predicate to accept matching signed
int32 or int64 storage; statistical and verdict rules were unchanged.

- [Original failed gate, byte-identical](evidence/s55__delivery_gate_s55.json)
- [Original failure receipt](harness/audit_history/original_FAILURE.json)
- [Original frozen analyzer](harness/audit_history/original_frozen_analyze.py)
- [Single-predicate diff](harness/audit_history/audit_predicate.diff)
- [Repair audit and regression disclosure](harness/AUDIT_REPAIR.json)
- [Complete repaired instrument gate](evidence/s55__delivery_gate_s55_dtype_repair.json)

The original failed gate is preserved byte-identical. The public analyzer includes
the disclosed correction for a **new** precommitment; it must not be substituted
into the historical S55 freeze. The original instrument did not pass.

## What passed, within the measured scope

Two-family prospective sign replication: **five independent seeds per arm** in
S54 and S55, with all four primary between-seed Student-t95 intervals excluding
zero in the declared directions. Generations within a seed are dependent.
Positive D_H means entropy loss; negative D_H means entropy gain.

| Experiment | Arm | Mean D_H (nats/generation) | Between-seed t95 interval |
| --- | --- | ---: | --- |
| S54 / Qwen2.5-7B | near_warm | -0.0537760002 | [-0.0572710198, -0.0502809807] |
| S55 / Mistral-7B | near_warm | -0.0566458836 | [-0.0766181280, -0.0366736392] |
| S54 / Qwen2.5-7B | near_cool | +0.0473138070 | [+0.0335535725, +0.0610740415] |
| S55 / Mistral-7B | near_cool | +0.0546631442 | [+0.0334714814, +0.0758548070] |

Both complete S55 postinitial mean entropy curves were inside their frozen
transferred envelopes. These empirical envelopes do not have guaranteed
new-family coverage. Matched no-training controls had exact-zero trained-loop
entropy drift. Repeated control losses reuse a verified fixed-model baseline;
they are not independent forward-pass replications. Initial entropy rounding of
2.22e-16 affects exact zero-width containment; the original control tolerance
passes and remains disclosed in the [measurement paper](docs/MODEL_COLLAPSE_MEASUREMENT_PAPER.md).

Fresh-dose drift cancellation was **measured at one operating point, 3 seeds**:
mean D_H changed from +0.138139613 to +0.00334192088 nats/generation
(rounded +0.138 -> +0.003). The fresh-arm interval includes zero; this is a
measured reduction toward zero, not exact cancellation or established prevention
against confirmed collapse. The warmer-arm quality result is **unresolved in both
directions**. Entropy stabilization is not quality preservation. S55 had no death arm.

## What failed

The S54 **collapse endpoint, timing band and collapse curve envelope failed**.
None of the three collapse seeds met the required two-consecutive-generation
confirmation by generation 15. Seed 2 crossed at generations 13 and 15 but
rebounded above the threshold at generation 14. Intervention holding therefore
did not establish prevention against a confirmed matched collapse. The earlier
Path-A "~90%, in flight" estimate was itself falsified.

The detailed [measurement paper](docs/MODEL_COLLAPSE_MEASUREMENT_PAPER.md)
retains those failures, the retrospective H1/H2 adjudication, and the distinction
between entropy-effective size and actual sample support.

## Lean audit scope

The historical S54/S55 release union contains **88 declarations: 80 selected
plus 8 S55**, checked against the standard axioms `propext`, `Classical.choice`
and `Quot.sound`, with **zero sorry**. This is a source-bound union, not an audit
of every theorem in the repository. Finite certificates and conditional
identities do not establish an unconditional transformer drift or collapse law.
See the [80-declaration gate](evidence/s54__corpus_kernel_gate.json),
[8-declaration gate](evidence/s55__kernel__kernel_gate.json) and
[theory manuscript](docs/MODEL_COLLAPSE_THEORY_CORPUS.md).

The later S56 restricted saturated categorical result has its own 132-declaration
combined audit, including 44 new declarations. It is separate from the frozen
88-declaration release scope. Its generic neural extension remains motivated,
unproven; the Q/V experiments do not instantiate the required categorical process.
No Lean source or experiment was changed for this publication.

## Reproduction and archive integrity

The validated original ZIP has **22 files, 82,097 bytes** and SHA-256:

`91a7e914356d74bd31a7264c3f6045a8eebad30923c32ad8abba4dea8fb7923f`

All original entries are preserved byte-identically under [`harness/`](harness/).
The original ZIP is attached to [release v0.4.1](https://github.com/aqida-ai/temperature-drift-harness/releases/tag/v0.4.1).
Root documentation and licenses are a separate publication wrapper. The archived
manifest's null license and unpublished flags describe its historical build;
they are deliberately preserved. The present license grant is in [LICENSE](LICENSE)
and [LICENSE-DOCS](LICENSE-DOCS).

The **fresh-folder CPU self-test passes 39/39**, using artificial software
fixtures and no model downloads or CUDA initialization. The separate storage
regression checks the disclosed dtype predicate. Read the saved receipts in
[`evidence/`](evidence/README.md). From a fresh checkout, with the required
NumPy/PyTorch packages available:

```text
cd harness
python -m public_harness.self_test --output cpu_self_test.json
python -m public_harness.storage_dtype_self_test --output storage_check.json
```

See [the original harness instructions](harness/README.md) for pinned public
model/data revisions, environment scope, input preparation and new precommitments.
**No model weights, selected corpus text, generated token pools, raw conditional
arrays or private data are included.** This is source plus metadata and selected
evidence; the full raw experiment archive is not included and cannot be reconstructed
from summary gates alone. The two explicitly released manuscripts use CC BY 4.0;
code and protocol/audit metadata use Apache-2.0. Third-party assets retain their
own licenses.

`RELEASE_MANIFEST.json` records per-file SHA-256 values. Git tree object IDs and
ZIP SHA-256 are different representations: verify the downloaded original ZIP's
SHA-256, all 22 extracted file hashes, and the corresponding Git subtree separately.
