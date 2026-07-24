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

if [[ ! -s "$COMMIT_FILE" ]]; then
  echo "missing deployed commit identity: $COMMIT_FILE" >&2
  exit 2
fi

docker build -f "$CONTROL/Dockerfile.v5-agent" -t "$IMAGE" "$CONTROL"

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  container_image="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
  if [[ "$container_image" != "$IMAGE" ]]; then
    echo "existing container uses unexpected image: $container_image" >&2
    exit 2
  fi
  docker start "$CONTAINER" >/dev/null
else
  docker run -d \
    --name "$CONTAINER" \
    --privileged \
    --restart unless-stopped \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home/wsx/AscendOpGenAgent:/home/wsx/AscendOpGenAgent \
    -v "$DATASET:$DATASET:ro" \
    -v "$CONTROL:$CONTROL" \
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
