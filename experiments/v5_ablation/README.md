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

Deploy the repository to `/home/wsx/AscendOpGenAgent`. Keep frozen experiment
assets in the sibling directory `/home/wsx/AscendOpGenAgent_assets`, with the
dataset at `cannbot_debug_inputs_n27_20260723` and this directory's control
files and frozen KB at `v5_ablation_control_20260724`. Write the deployed Git
commit to:

`/home/wsx/AscendOpGenAgent_assets/v5_ablation_control_20260724/V5_GIT_COMMIT.txt`

Then run `prepare_v5_runtime.sh`. It builds/starts the container, regenerates
fingerprints, verifies the actual model, proves all ablation profiles, runs the
27-task clean-build and initial-validation preflight across NPU 3-7, and makes
source/KB inputs read-only. Preparation also verifies that every frozen arm
manifest matches the executable profile and launch constants.
An explicit `507015`/`NPU_AICORE_EXCEPTION` preflight result is clean-built
again at most twice; every transient attempt is archived, and a stable
non-infrastructure result becomes the formal initial classification.

During a formal run, code/control-plane/source/KB fingerprints are checked
before and after every arm. Provider environment files live in an ephemeral
directory below `.secrets` and are removed by the supervisor. Each post-hoc
task is rebuilt with `build_ascendc.py --clean` before verification. After all seven arms,
`utils/analyze_v5_ablation.py` writes the paired success, final-valid-cycle
cost, long-failure, failed-cycle and arm-compliance closure package.

Preparation does not start a formal arm. `launch_v5_ablation.sh` remains locked
until
`/home/wsx/AscendOpGenAgent_assets/v5_ablation_control_20260724/ARMED_BY_USER.txt`
contains exactly `START_V5_ABLATION`.

See `docs/v5_ablation_experiment_master_design_20260724.md` for the experiment
contract and analysis rules.
