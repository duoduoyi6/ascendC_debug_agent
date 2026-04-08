#include "rms_norm_single_row_kernel.h"

extern "C" __global__ __aicore__ void rms_norm_single_row_custom(
    GM_ADDR x,
    GM_ADDR gamma,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    RmsNormSingleRowKernel kernel;
    kernel.Init(x, gamma, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void rms_norm_single_row_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *gamma,
    uint8_t *y,
    uint8_t *tiling)
{
    rms_norm_single_row_custom<<<blockDim, nullptr, stream>>>(x, gamma, y, tiling);
}
#endif
