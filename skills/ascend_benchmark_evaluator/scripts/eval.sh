#!/bin/bash
set -e  # 遇到错误立即退出

# 检查参数个数
if [ $# -ne 2 ]; then
    echo "用法: $0 <op_name> <dataset>"
    exit 1
fi

OP_NAME=$1
DATASET=$2

BASE_DIR="/home/lvyou/y00889327/OpenOps-main"
CUSTOM_PATH="${BASE_DIR}/output/${OP_NAME}/vendors/customize"

echo "=== 1. 查看 Available ops ==="
set +e  # 临时关闭错误退出，即使第一步出错也继续
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_PATH}"
source "${CUSTOM_PATH}/bin/set_env.bash"
python3 -c "import torch; import torch_npu; import custom_ops_lib; print('Available ops:', [a for a in dir(custom_ops_lib) if not a.startswith('_')])"
set -e  # 恢复错误退出

echo "=== 2. 重装 pybind ==="
cd "${BASE_DIR}"
python3 .opencode/skills/ascend_benchmark_evaluator/scripts/generate_pybind.py "${OP_NAME}"

echo "=== 3. 查看重装后 Available ops ==="
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_PATH}"
source "${CUSTOM_PATH}/bin/set_env.bash"
python3 -c "import torch; import torch_npu; import custom_ops_lib; print('Available ops:', [a for a in dir(custom_ops_lib) if not a.startswith('_')])"

echo "=== 4. 执行测评 ==="
python3 .opencode/skills/ascend_benchmark_evaluator/scripts/eval_operator_generic.py --op "${OP_NAME}" --dataset "${DATASET}"

echo "完成"
