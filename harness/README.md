<!--
AQIDA_CONSTITUTIONAL_SCOPE: baseline_or_comparator
AQIDA_ROLE: baseline_or_comparator
THEORY_BINDING_MAP: CORE_LOGIC: signed measurement boundary; elm_eulerian:
complete finite composition; n_s_u: frozen sources and independent evidence.
CLAIM_BOUNDARY: comparator only; not AQiDA core logic. Prospective experiment harness.
FALSIFICATION_TESTS: CPU self-test, source hashes, complete prearm/runtime gates.
-->
# Temperature-drift precommitment harness

**A reproducible instrument for experiments with frozen entropy paths, complete
seed trajectories, and separate held-out quality measurements.**

Read `VALIDATION_RECORD.json` for the exact scope of the completed example
and `PACKAGE_MANIFEST.json` for the files and hashes in this archive.
A successful instrument audit can accompany an unresolved or adverse
scientific outcome. The archive does not claim public priority or a
universal critical temperature.

This code prepares public text inputs, commits complete prospective entropy
curves and seed grids, runs recursive Q/V LoRA training, and records held-out
next-token quality beside signed entropy drift. It is a classical transformer
comparator. Its scalar measurements are not an AQiDA phase-reasoning mechanism.

The retained second-setting experiment uses
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71)
and the [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext).
The referenced model and dataset retain their own licenses. Model weights and
corpus text are not distributed in the code archive.

## What the instrument measures

For fixed prompts, `D_H = H_source - H_production`; positive values denote an
entropy decrease. Entropy is calculated over the complete model vocabulary and
then averaged uniformly over prompts. It is not a quality score or a literal
count of nonzero vocabulary entries.

Training supervises generated continuations only. A 64-token prompt followed by
128 generated tokens supplies 128 optimizer targets per example. The final
prompt-position logit predicts the first continuation token. Prompt tokens still
condition predictions; their own next-token labels are excluded from the loss.
The adapter is reset to the same per-trajectory initial A/B factors before each
generation's production fit. The preceding fitted adapter generates the next
training pool. Sampling continues for the declared fixed number of tokens even
when an end-of-sequence token is sampled. No top-k or nucleus filter is applied.

Quality is teacher-forced negative log likelihood on a fixed set of separate
test continuations, using raw model predictions at beta=1. The aggregate divides
the total negative log probability by the exact count of evaluated target
tokens. Perplexity is the exponential of that mean. This is selected-window
perplexity, not full-corpus WikiText benchmark perplexity. Compare changes within
one tokenizer; do not compare raw perplexity values across tokenizers. See the
[Transformers explanation of fixed-context perplexity](https://huggingface.co/docs/transformers/en/perplexity)
for the dependence on evaluation context and tokenization.

Distinct-2 is the number of unique token-ID bigrams divided by all bigram
positions in the generated continuations. It excludes prompts and never creates
bigrams across sequence boundaries. It measures token diversity, not semantic
quality.

For generation g, distinct-2 describes the incoming training pool sampled
from the generation-(g-1) teacher at the arm beta, before fitting. Held-out
NLL describes the newly fitted generation-g adapter. Generation-five
distinct-2 therefore uses the generation-four teacher; the final fitted
adapter receives no additional diversity sample. The metrics have different
measurement stages.

## Package layout

- `prepare_wikitext.py`: exact source revisions, tokenizer checks, fixed training
  prompts, separate test windows and source offsets.
- `precommitment.py`: finite-path and seed checks, immutable source hashes, and
  exact committed-byte verification.
- `transfer_forecast.py`: transfer an already frozen entropy path to a new
  initial entropy and vocabulary scale, without refitting response outcomes.
- `metrics.py`: causal labels, exact target counts, held-out NLL/perplexity and
  continuation-only distinct-n.
- `engine.py`: normalized full-vocabulary sampling, Q/V LoRA, reset fitting,
  checkpoints and conditional-distribution measurements.
- `run.py`: resumable generation stages and complete evidence records.
- `launch.py`: an exclusive run lock, 60 seconds of idle CUDA before launch, and
  resource observations every 15 seconds. If a foreign CUDA process appears,
  it stops only the worker it launched.
- `self_test.py`: an entirely artificial CPU example that requires no model
  downloads, local AQiDA repository, or GPU.
- `analyze.py`: a fixed five-generation analysis profile that verifies the
  complete saved grid and computes uncertainty between independent seeds.

## CPU verification

Run commands from the directory containing the `public_harness` folder. The CPU
self-test requires NumPy and PyTorch, and initializes no CUDA device.

```text
python -m public_harness.self_test --output cpu_self_test.json
```

The test's entropy values and tiny model are software fixtures. They must never
be reused as a scientific forecast or treated as a model-scale experiment.

## Prepare real inputs

Cache the pinned public tokenizer/configuration and WikiText files using the
Hugging Face CLI. Input preparation uses cached files only; it does not silently
download model weights or start inference.

```text
hf download mistralai/Mistral-7B-Instruct-v0.3 config.json tokenizer.json tokenizer_config.json special_tokens_map.json --revision c170c708c41dac9275d15a8fff4eca08d52bab71
hf download Salesforce/wikitext wikitext-2-raw-v1/train-00000-of-00001.parquet wikitext-2-raw-v1/test-00000-of-00001.parquet --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3
python -m public_harness.prepare_wikitext --model mistralai/Mistral-7B-Instruct-v0.3 --model-revision c170c708c41dac9275d15a8fff4eca08d52bab71 --outdir prepared_inputs
```

The default preparation selects 128 unique 64-token prompts from train and 128
disjoint test windows of 64 prompt tokens plus 128 target tokens. Source rows are
joined with two newlines, tokenized without chat formatting or added special
tokens, and sampled by the recorded RNG seeds. Full selected test windows that
also occur verbatim in the training stream are excluded before model evaluation.
Shorter overlaps and base-pretraining exposure remain unknown. The test targets
must never enter calibration, forecast selection or optimizer training.

The pinned Transformers 4.50.2 Mistral loader needs `add_prefix_space=None` to
avoid a slow-tokenizer conversion path. This retains the normalization in the
published fast-tokenizer file. Preparation compares token IDs against that file
on fixed whitespace, multilingual, code and source-text probes. A different
library/tokenizer combination requires a new verified protocol.

## Freeze before any arm

Create a numerical `design.json` from a separately documented calibration or
transfer model, and an explicit `files.json` list of relative POSIX source paths.
The runner requires the full seed grid, exact model/data revisions, every
generation's central/lower/upper entropy path, the uncertainty unit, the complete
horizon, quality definition, and predeclared interpretation rules. It checks
that entropy paths remain in `[0, log(V)]` and that their adjacent differences
telescope. This does not certify forecast accuracy or confidence-band coverage.

The runtime configuration also names prepared arrays and their manifest, a
previously saved base `log_probabilities` array, exact package versions, adapter
dimensions, training budget, resource guards and replay tolerances. This source
package does not invent those values for a user. They must be established and
recorded before the arms. The `validate_runtime_config` function lists the
required runtime fields. Calibration and arm RNG seeds must be disjoint.

For a prospective transfer, `transfer_paths` translates complete normalized-
entropy logit paths by the difference between the old and new initial coordinates.
It retains the source temperatures, reconstructs every adjacent entropy drift,
and rejects invalid initial values. Its decision envelope is a transferred
empirical hypothesis; it has no automatic confidence-coverage guarantee for the
new setting. Record the transfer rule before observing the new base, and compile
the numerical paths before evaluating any arm or held-out quality outcome.

The optional `snapshot_sha256` runtime field maps each pinned model filename to
its `sha256` and `bytes`. When present, the engine verifies every listed cached
file before loading the model, using the exact revision and local files only.
Bind all metadata and weight shards recorded by the untouched-base observation.
The ordinary base and checkpoint log-probability replay checks still apply.

Freeze all executable files, input arrays/manifests, base-calibration evidence,
forecast producers, and the environment/decision records needed to reproduce
the design. Protect every package module, including the supervisor. Store the
commitment and its source bytes in Git with `-text` attributes so exact-byte
checking also works on Windows.

```text
python -m public_harness.precommitment validate --root . --design design.json
python -m public_harness.precommitment freeze --root . --design design.json --files files.json --commitment precommitment.json
```

Commit the explicit source/input list and `precommitment.json` before any arm.
Then check the committed bytes and complete runtime configuration:

```text
python -m public_harness.precommitment verify --root . --commitment precommitment.json --require-committed
python -m public_harness.run --root . --precommitment precommitment.json --output run --check-only
```

Filesystem dates alone do not establish public preregistration. Publish or
archive the exact commitment hash before the arms when that stronger provenance
is part of the study. This package does not publish files automatically.

## Execute and resume

After the actual model weights and pinned runtime are available, use the
supervisor to enforce the single-worker discipline:

```text
python -m public_harness.launch --root . --precommitment precommitment.json --output run
```

An isolated package directory can be supplied through `--runtime-packages`; it
must contain the exact versions recorded in the protocol. Do not update the
environment during a running experiment. Plan disk space for the weights,
checkpoints, saved full-vocabulary conditional arrays, and restart headroom.

Create `run/PAUSE_AFTER_GENERATION` to request a resource pause at the next
completed-generation checkpoint. Remove that exact marker before resuming with
the same launch command. The fixed scientific horizon remains unchanged. An
orphaned checkpoint or samples without the expected saved hash cause a failure
that must be audited, rather than silently accepted or replaced.

Each completed generation retains its tokens, training indices and losses,
adapter, conditional log probabilities, per-prompt entropy accounting, quality
rows and timing scope. Resume checks compare saved log probabilities and hashes,
including evidence from earlier completed generations. Timing fields distinguish
work in the current attempt from persisted sampling/training phases.

## Analysis and release boundary

The runner never converts partial observations into significance or collapse
claims. `execution_complete.json` means only that execution reached the declared
horizon. Independent analysis must still verify the entire grid, the original
forecast/decision rules, between-seed uncertainty and all claim gates.

`analyze.py` supplies a fixed analysis profile: five generations, five independent
seeds in each of `near_warm` and `near_cool`, and three paired `no_training`
controls. Include the analyzer itself in the prearm source list. Its primary
test requires both seed-level Student-t95 intervals to exclude zero in their
declared directions. It reports complete-curve agreement and held-out NLL
separately. A partial grid produces no statistical verdict; a valid completed
run can have a negative scientific result.

```text
python -m public_harness.analyze --root . --precommitment precommitment.json --run run --output analysis.json
```

The CPU analysis independently reconstructs entropy from saved conditional
arrays, every prefix telescope, decoder diagnostics, exact training schedules
and label counts, continuation distinct-2, and token-weighted held-out NLL from
saved per-example losses. It checks all sample/checkpoint hashes and teacher
links, including early generations, and verifies that every recorded launch
commit already contained the exact precommitment and its sources. It remains
on CPU. It does not independently reevaluate the model's held-out logits.
Local Git provenance is not a public preregistration timestamp.

The transferred envelopes have no new-family confidence-coverage guarantee.
Held-out NLL may worsen even when drift signs resolve. An interval containing
zero does not establish quality equivalence. Distinct-2 is a token diversity
diagnostic and is allowed to vary in the fixed-model resampling control.

The archive preserves the executable sources used by the retained example.
`VALIDATION_RECORD.json` reports the actual complete instrument outcome,
scientific verdict, quality outcomes, precommitment hash, and local commit
provenance. CPU self-tests remain artificial software evidence. The
independent arithmetic audit does not supply an additional model forward
pass, and no-training losses reuse the verified fixed-model baseline.

This is a code-and-metadata release: model weights, selected corpus text,
raw conditional arrays, generated samples, and private theory documents
are absent. The validation record is a hash-bound summary, not the full
raw experiment archive. Use the pinned official model/data sources and
prepare a new auditable commitment to run another study. The supplied
`S55_PROTOCOL.json` documents the original design; its original input
paths and hashes do not make it directly runnable from this code archive.

The code license and publication status are stated explicitly in the
package manifest. A local archive is not evidence that it has been
published or that a public preregistration occurred before the arms.

## Disclosed storage-dtype repair

The original frozen analyzer failed before computing aggregate statistics: it required int64 on-disk tokens although the precommitted preparation producer saved int32. The prepared context and held-out arrays match their original precommitment hashes exactly. The original FAIL_INSTRUMENT receipt and analyzer remain preserved. A separately committed post-run repair changes exactly one storage-width predicate to accept matching signed int32 or int64; all original shape, range, probability, provenance, teacher-chain, sample-budget, seed-interval and verdict rules remain in force. Twenty focused regression checks preceded the repaired aggregate analysis. This is a post-run software-repaired instrument result under unchanged predeclared statistical rules; the original frozen instrument did not pass.

The executable `public_harness/analyze.py` in this archive contains that one predicate correction for future studies. Include it in a NEW precommitment before running any arms. Do not substitute it into the original S55 commitment: the original failed source and failure receipt are preserved in `audit_history`, and `AUDIT_REPAIR.json` records the exact source difference and historical audit provenance. Reproducing the historical full audit additionally needs the original full raw archive and its separately committed repair wrapper. This package contains source and metadata, not that raw archive.

Input preparation is the exact prearm producer bound by its SHA in the input manifest; it was not itself a top-level source-file entry in the original S55 commitment. Freeze it explicitly in future studies.

Run the additional artificial storage regression with `python -m public_harness.storage_dtype_self_test --output storage_check.json`. It inspects the exported analyzer predicate directly and uses no model or corpus.
