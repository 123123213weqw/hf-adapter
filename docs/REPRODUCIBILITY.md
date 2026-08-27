# Reproducibility artifacts

Local files are the source of truth even when W&B reporting is enabled.

Each evaluation or training run writes:

- resolved configuration and exact command;
- source Git SHA, model revision and dataset revision;
- Python, PyTorch, Transformers, TRL and PEFT versions;
- JSONL metrics and final evaluation;
- checkpoint inventory and SHA256;
- stdout/stderr paths and exit status;
- optional W&B run ID and URL, never a token.

Release bundles live below `results/`. Large task sample logs can be
attached to a release artifact while the manifest and summary stay in Git.
Committed result summaries must be immutable and must identify the exact GPU.

The performance history is intentionally not mixed with the reference line. It
is retained on `perf/native-kernels-v0.8`.

Candidate optional-backend evidence is stored below
`results/native-backend/`. A compact committed bundle may contain only gate
summaries, environment and revision metadata, metric validators, resolved
training configuration, and SHA256 manifests. It must not contain model
weights, adapter weights, task samples, W&B run directories, checkpoints, or
large stdout/stderr logs.

Infrastructure corrections must be additive and auditable: retain the original
status, write a `final-corrected-exit-status.json`, and list the failed attempt,
cause, replacement evidence, and remaining numerical failures. Never rewrite
an old failed status into a pass.
