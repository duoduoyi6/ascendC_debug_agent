#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/wsx/AscendOpGenAgent
ASSETS_ROOT=/home/wsx/AscendOpGenAgent_assets
CONTROL="$ASSETS_ROOT/v5_ablation_control_20260724"
DATASET="$ASSETS_ROOT/cannbot_debug_inputs_n27_20260723"
KB="$CONTROL/initial_kb_102_cleaned_20260715.json"
IMAGE=ascendc-v5-agent:20260724
CONTAINER=v5_cann
COMMIT_FILE="$CONTROL/V5_GIT_COMMIT.txt"

mkdir -p "$ROOT/outputs" "$ROOT/.secrets"

if [[ ! -s "$COMMIT_FILE" ]]; then
  echo "missing deployed commit identity: $COMMIT_FILE" >&2
  exit 2
fi

docker build -f "$CONTROL/Dockerfile.v5-agent" -t "$IMAGE" "$CONTROL"

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    container_image="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
    container_image_id="$(docker inspect -f '{{.Image}}' "$CONTAINER")"
    expected_image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
    root_rw="$(docker inspect -f \
      "{{range .Mounts}}{{if eq .Destination \"$ROOT\"}}{{.RW}}{{end}}{{end}}" \
      "$CONTAINER")"
    outputs_rw="$(docker inspect -f \
      "{{range .Mounts}}{{if eq .Destination \"$ROOT/outputs\"}}{{.RW}}{{end}}{{end}}" \
      "$CONTAINER")"
    dataset_rw="$(docker inspect -f \
      "{{range .Mounts}}{{if eq .Destination \"$DATASET\"}}{{.RW}}{{end}}{{end}}" \
      "$CONTAINER")"
    control_rw="$(docker inspect -f \
      "{{range .Mounts}}{{if eq .Destination \"$CONTROL\"}}{{.RW}}{{end}}{{end}}" \
      "$CONTAINER")"
    if [[ "$container_image" != "$IMAGE" \
          || "$container_image_id" != "$expected_image_id" \
          || "$root_rw" != "false" \
          || "$outputs_rw" != "true" \
          || "$dataset_rw" != "false" \
          || "$control_rw" != "false" ]]; then
      echo "recreating $CONTAINER to enforce the frozen mount/image contract"
      docker rm -f "$CONTAINER" >/dev/null
    fi
fi

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  docker start "$CONTAINER" >/dev/null
else
  docker run -d \
    --name "$CONTAINER" \
    --privileged \
    --restart unless-stopped \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v "$ROOT:$ROOT:ro" \
    -v "$ROOT/outputs:$ROOT/outputs" \
    -v "$DATASET:$DATASET:ro" \
    -v "$CONTROL:$CONTROL:ro" \
    "$IMAGE"
fi

docker exec "$CONTAINER" bash -lc \
  'python3 -c "import torch, torch_npu, numpy; print(torch.__version__, torch_npu.__version__, numpy.__version__)"'
docker exec "$CONTAINER" bash -lc 'claude --version'
docker exec "$CONTAINER" bash -lc 'npu-smi info'

python3 "$CONTROL/prepare_v5_manifests.py" \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --kb "$KB" \
  --control "$CONTROL" \
  --git-commit "$(tr -d '\r\n' < "$COMMIT_FILE")"

chmod 0755 "$CONTROL/claude_qwen38max.sh"
chmod 0755 "$CONTROL/launch_v5_ablation.sh"
chmod 0755 "$CONTROL/smoke_qwen38max_provider.sh"
chmod 0600 "$ROOT/.secrets/v5_qwen38max_provider.json"

"$CONTROL/smoke_qwen38max_provider.sh"

python3 "$ROOT/utils/smoke_v5_ablation_profiles.py" \
  --repo-root "$ROOT" \
  --output "$CONTROL/ablation_profile_smoke.json" \
  --kb-path "$KB"

python3 "$ROOT/utils/verify_v5_arm_contracts.py" \
  --repo-root "$ROOT" \
  --control "$CONTROL" \
  --kb "$KB" \
  --report "$CONTROL/arm_contract_verification.json"

if [[ -e "$CONTROL/dataset_preflight" ]]; then
  echo "refuse existing dataset preflight: $CONTROL/dataset_preflight" >&2
  exit 2
fi
python3 "$ROOT/utils/run_v5_dataset_preflight.py" \
  --repo-root "$ROOT" \
  --source-root "$DATASET/tasks" \
  --output "$CONTROL/dataset_preflight" \
  --container "$CONTAINER" \
  --npus 3,4,5,6,7 \
  --tilelang-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --timeout 43200 \
  --expected-tasks 27

chmod -R a-w "$DATASET"
chmod a-w "$KB"

echo "V5 runtime prepared. Formal experiment remains locked."
