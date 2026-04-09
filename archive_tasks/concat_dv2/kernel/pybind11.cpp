#include <algorithm>
#include <array>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "concat_dim0_tiling.h"

extern "C" void concat_dim0_1_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x0,
    uint8_t *x1,
    uint8_t *x2,
    uint8_t *x3,
    uint8_t *y,
    uint8_t *tiling);
extern "C" void concat_dim0_2_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x0,
    uint8_t *x1,
    uint8_t *x2,
    uint8_t *x3,
    uint8_t *y,
    uint8_t *tiling);
extern "C" void concat_dim0_3_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x0,
    uint8_t *x1,
    uint8_t *x2,
    uint8_t *x3,
    uint8_t *y,
    uint8_t *tiling);
extern "C" void concat_dim0_4_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x0,
    uint8_t *x1,
    uint8_t *x2,
    uint8_t *x3,
    uint8_t *y,
    uint8_t *tiling);

namespace current_task_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

int32_t ChooseBlockM(int32_t totalM)
{
    for (int32_t candidate : {64, 32, 16, 8, 4, 2}) {
        if (totalM >= candidate) {
            return candidate;
        }
    }
    return 2;
}

void CheckInputTensor(const at::Tensor &tensor, const char *name, const at::Tensor &reference)
{
    TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device().type() == reference.device().type(), name, " must be on the same device as x0");
    TORCH_CHECK(tensor.sizes()[1] == reference.sizes()[1], name, " must have the same trailing width as x0");
}

uint8_t *DataPtr(const at::Tensor &tensor)
{
    return static_cast<uint8_t *>(const_cast<void *>(tensor.storage().data()));
}

at::Tensor RunConcatDim0(const std::array<const at::Tensor *, 4> &inputs, int32_t inputCount, LaunchFn launch)
{
    TORCH_CHECK(inputCount >= 1 && inputCount <= 4, "inputCount must be in [1, 4]");

    const at::Tensor &x0 = *inputs[0];
    CheckInputTensor(x0, "x0", x0);
    TORCH_CHECK(x0.device().type() == at::kPrivateUse1, "x0 must be on NPU/PrivateUse1 device");

    int32_t m[4] = {
        static_cast<int32_t>(x0.sizes()[0]),
        0,
        0,
        0,
    };
    for (int32_t index = 1; index < inputCount; ++index) {
        const at::Tensor &tensor = *inputs[index];
        const std::string name = "x" + std::to_string(index);
        CheckInputTensor(tensor, name.c_str(), x0);
        m[index] = static_cast<int32_t>(tensor.sizes()[0]);
    }

    const int32_t n = static_cast<int32_t>(x0.sizes()[1]);
    const int32_t totalM = m[0] + m[1] + m[2] + m[3];
    at::Tensor y = at::empty({static_cast<int64_t>(totalM), static_cast<int64_t>(n)}, x0.options());
    if (totalM == 0 || n == 0) {
        return y;
    }

    const int32_t blockM = ChooseBlockM(totalM);
    const int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    const int32_t blockNum = CeilDivI32(totalM, blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, blockNum);
    const int32_t tasksPerCore = CeilDivI32(blockNum, usedCoreNum);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(ConcatDim0Tiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<ConcatDim0Tiling *>(tilingCpu.data_ptr());
    tiling->M0 = m[0];
    tiling->M1 = m[1];
    tiling->M2 = m[2];
    tiling->M3 = m[3];
    tiling->inputCount = inputCount;
    tiling->N = n;
    tiling->totalM = totalM;
    tiling->blockM = blockM;
    tiling->subBlockM = subBlockM;
    tiling->blockNum = blockNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        DataPtr(x0),
        inputCount >= 2 ? DataPtr(*inputs[1]) : nullptr,
        inputCount >= 3 ? DataPtr(*inputs[2]) : nullptr,
        inputCount >= 4 ? DataPtr(*inputs[3]) : nullptr,
        DataPtr(y),
        DataPtr(tilingNpu));
    return y;
}

at::Tensor run_concat_dim0_1(const at::Tensor &x0)
{
    return RunConcatDim0({&x0, nullptr, nullptr, nullptr}, 1, concat_dim0_1_do);
}

at::Tensor run_concat_dim0_2(const at::Tensor &x0, const at::Tensor &x1)
{
    return RunConcatDim0({&x0, &x1, nullptr, nullptr}, 2, concat_dim0_2_do);
}

at::Tensor run_concat_dim0_3(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2)
{
    return RunConcatDim0({&x0, &x1, &x2, nullptr}, 3, concat_dim0_3_do);
}

at::Tensor run_concat_dim0_4(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &x3)
{
    return RunConcatDim0({&x0, &x1, &x2, &x3}, 4, concat_dim0_4_do);
}

}  // namespace current_task_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "current_task concat_dim0 AscendC extension";
    m.def("run_concat_dim0_1", &current_task_ext::run_concat_dim0_1, "");
    m.def("run_concat_dim0_2", &current_task_ext::run_concat_dim0_2, "");
    m.def("run_concat_dim0_3", &current_task_ext::run_concat_dim0_3, "");
    m.def("run_concat_dim0_4", &current_task_ext::run_concat_dim0_4, "");
}
