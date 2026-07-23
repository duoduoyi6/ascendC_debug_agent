# V5 Ablation Control

This directory contains the reproducible, non-secret control plane for the
27-task V5 ablation experiment on `101.245.78.76`.

## Frozen treatment

- Arms: `full`, `no_kb`, `no_diagnostic_evidence`, `no_loopguard`,
  `no_anticheat`, `no_fulleval`, `baseline`
- Model: `qwen3.8-max-preview`
- Provider mode: one fixed provider, five concurrent workers, no quota query
- NPUs: `3,4,5,6,7`
- Task budgets: `max_attempts=5`, `max_turns=240`,
  `soft_task_turns=480`, `max_task_turns=600`
- KB: the included 102-entry snapshot, read-only when enabled

The API credential is intentionally absent. On the server it must exist only at
`/home/wsx/AscendOpGenAgent/.secrets/v5_qwen38max_provider.json` with mode
`0600`.

## Preparation

Deploy the repository to `/home/wsx/AscendOpGenAgent`, copy this directory's
control files and frozen KB to `/root/v5_ablation_control_20260724`, and write
the deployed Git commit to:

`/root/v5_ablation_control_20260724/V5_GIT_COMMIT.txt`

Then run `prepare_v5_runtime.sh`. It builds/starts the container, regenerates
fingerprints, verifies the actual model, proves all ablation profiles, runs the
27-task initial-validation preflight across NPU 3-7, and makes source/KB inputs
read-only.

Preparation does not start a formal arm. `launch_v5_ablation.sh` remains locked
until `/root/v5_ablation_control_20260724/ARMED_BY_USER.txt` contains exactly
`START_V5_ABLATION`.

See `docs/v5_ablation_experiment_master_design_20260724.md` for the experiment
contract and analysis rules.
