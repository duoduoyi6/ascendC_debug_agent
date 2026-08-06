---
name: cannbench-hidden-followup
description: Request and track owner-authorized CANNBench hidden evaluation after a standard job scores strictly above 50.
---

# CANNBench Hidden Follow-up

Use this skill only from the remote experiment repository. It uses the official
CANNBench API and the server-side secret file; never print credentials or copy
them into an experiment artifact.

## Eligibility check

```bash
bash utils/run_cannbench_hidden_followup.sh \
  --standard-job job_STANDARD \
  --evidence-root outputs/EXPERIMENT/hidden_followup
```

The standard job must be terminal, use the standard case set, retain its
submission ID and selected operators, and have `score > 50`. A score of exactly
50 is not eligible.

## Request and monitor

```bash
bash utils/run_cannbench_hidden_followup.sh \
  --standard-job job_STANDARD \
  --evidence-root outputs/EXPERIMENT/hidden_followup \
  --request-hidden --poll
```

If a hidden job was started in the website, bind and monitor it without creating
a duplicate:

```bash
bash utils/run_cannbench_hidden_followup.sh \
  --standard-job job_STANDARD \
  --hidden-job job_HIDDEN \
  --evidence-root outputs/EXPERIMENT/hidden_followup \
  --poll
```

The evidence root records eligibility, the request response, standard/hidden
job binding, terminal result, owner-visible logs, and downloadable artifacts.

## Failure analysis boundary

Use only fields and artifacts returned to the submission owner. Preserve case
IDs, shapes, dtypes, error metrics, compile/runtime logs, and anti-cheat details
when the API exposes them. If exact hidden inputs are withheld, create local
surrogate stress cases from the observable failure signature and vary one repair
mechanism at a time. Do not call administrator-only endpoints, scrape session
cookies, or claim that inferred stress cases are the platform's hidden inputs.
