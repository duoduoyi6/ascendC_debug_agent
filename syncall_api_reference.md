# SyncAll — Inter-Core Synchronization API Reference
**Source:** https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0204.html  
**Product:** CANN Commercial Edition 8.0.0 — Ascend C Operator Development API

---

## Function Usage

When different AI Cores operate the same global memory block, this function can be called to synchronize the AI Cores to avoid data dependency problems such as write-after-read, read-after-write, and write-after-write.

Currently, multi-core synchronization is classified into two types:
- **Hardware synchronization**: uses the full-core synchronization instruction of the hardware to ensure multi-core synchronization.
- **Software synchronization**: implemented through software algorithm simulation.

---

## Prototype

### Soft Synchronization

```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(
    const GlobalTensor<int32_t>& gmWorkspace,
    const LocalTensor<int32_t>& ubWorkspace,
    const int32_t usedCores = 0
)
```

### Hard Synchronization

```cpp
template<bool isAIVOnly = true>
__aicore__ inline void SyncAll()
```

---

## Parameters

| Parameter     | Input/Output | Description |
|---------------|--------------|-------------|
| `gmWorkspace` | Input | User-defined global space serving as the cache shared by all cores. Used to store the status flag of each core. Type: `GlobalTensor<int32_t>`. **Not supported in the hardware synchronization API.** For required space and precautions, see Constraints below. |
| `ubWorkspace` | Input | User-defined local space. Used by each core independently to mark the status of the current core. Type: `LocalTensor<int32_t>`. Supported TPosition: `VECIN`, `VECCALC`, `VECOUT`. **Not supported in the hardware synchronization API.** For required space, see Constraints below. |
| `usedCores`   | Input | Specifies the number of cores to be synchronized. The input value cannot exceed the logical `blockDim` value specified during operator calling. If not passed in, full-core soft synchronization is enabled. **Supported only in the soft synchronization API.** |
| `isAIVOnly`   | Input | Indicates whether synchronization is performed only between vector cores. Default value: `true`. To enable MIXCORE, set to `false`. |

---

## Returns

None

---

## Availability

- **Soft synchronization**: Atlas Training Series Product
- **Hard synchronization**: (see official documentation for supported hardware list)

---

## Constraints

1. The size of the `gmWorkspace` cache must be **≥ (number of cores × 32 bytes)**, and the cache value must be **initialized to 0**. Two common initialization modes:
   - Perform initialization on the host to ensure `gmWorkspace` has been initialized to 0 before this API is called.
   - Initialize the `gmWorkspace` cache during kernel initialization. Note that **all** `gmWorkspace` cache space needs to be initialized on each core.

2. The size of the space allocated for `ubWorkspace` must be **≥ (number of cores × 32 bytes)**.

3. Currently, the **hardware synchronization API cannot be used in the kernel launch project** — it can only be used in the custom operator project. In addition, the workspace size in the Tiling function cannot be set to 0.

4. When this API is used for multi-core control, the logical `blockDim` specified during operator calling must be **≤ the number of cores** for running the operator. Otherwise, the framework inserts abnormal synchronization during multi-round scheduling, causing the kernel to stop responding.

---

## Example

In this example, eight cores are used for data processing. Each core processes 32 pieces of `float`-type data. The data is multiplied by 2 and then added to the data on other cores that are multiplied by 2 in the same way. The intermediate result is saved to `workGm`. Therefore, data synchronization between multiple cores is required.

> **Note:** When software synchronization is used, the `syncGm` value passed by the entrypoint function must have been initialized to 0 on the host. If hardware synchronization is used, `syncGm` and `workQueue` do not need to be passed.

```cpp
#include "kernel_operator.h"

const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

class KernelSyncAll {
public:
    __aicore__ inline KernelSyncAll() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* workGm,
        __gm__ uint8_t* syncGm)
    {
        blockNum = AscendC::GetBlockNum();           // Obtain the total number of cores.
        perBlockSize = srcDataSize / blockNum;        // Each core evenly processes the same number of pieces of data.
        blockIdx = AscendC::GetBlockIdx();           // Obtain the ID of the current working core.
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workGm), srcDataSize);
        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncGm), blockNum * DEFAULT_SYNCALL_NEED_SIZE);
        pipe.InitBuffer(inQueueSrc1, 1, perBlockSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc2, 1, perBlockSize * sizeof(float));
        pipe.InitBuffer(workQueue, 1, blockNum * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));
        pipe.InitBuffer(outQueueDst, 1, perBlockSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        FirstCompute();
        CopyToWorkGlobal();  // Save the data computed by the current working core to the external workspace.
        // Wait until all cores complete the computation.
        AscendC::LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();
        AscendC::SyncAll(syncGlobal, workLocal);
        workQueue.FreeTensor(workLocal);
        // The final addition result needs to be computed after computation on all cores are complete.
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        AscendC::DataCopy(dstLocal, srcLocal2, perBlockSize);  // Save the data computed by the current working core to the destination space.
        inQueueSrc2.FreeTensor(srcLocal2);
        for (int i = 0; i < blockNum; i++) {
            if (i != blockIdx) {
                CopyFromOtherCore(i);  // Read data from the external workspace.
                Accumulate(dstLocal);  // All data is added to the destination space.
            }
        }
        outQueueDst.EnQue(dstLocal);
        CopyOut();
    }
private:
    __aicore__ inline void CopyToWorkGlobal()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(workGlobal[blockIdx * perBlockSize], dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyFromOtherCore(int index)
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, workGlobal[index * perBlockSize], perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }
    __aicore__ inline void Accumulate(const AscendC::LocalTensor<float> &dstLocal)
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::Add(dstLocal, dstLocal, srcLocal1, perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
    }
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, srcGlobal, perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }
    __aicore__ inline void FirstCompute()
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.AllocTensor<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        float scalarValue(2.0);
        AscendC::Muls(dstLocal, srcLocal1, scalarValue, perBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(srcLocal2, dstLocal, perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.EnQue(srcLocal2);
        outQueueDst.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc2;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> workQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> srcGlobal;
    AscendC::GlobalTensor<float> dstGlobal;
    AscendC::GlobalTensor<float> workGlobal;
    AscendC::GlobalTensor<int32_t> syncGlobal;
    int srcDataSize = 256;
    int32_t blockNum = 0;
    int32_t blockIdx = 0;
    uint32_t perBlockSize = 0;
};

extern "C" __global__ __aicore__ void kernel_syncAll_float(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm,
    __gm__ uint8_t* workGm, __gm__ uint8_t* syncGm) {
    KernelSyncAll op;
    op.Init(srcGm, dstGm, workGm, syncGm);
    op.Process();
}
```

**Expected output:**
```
Input  (srcGm): [1,1,1,1,1,...,1]
Output (dstGm): [16,16,16,16,16,...,16]
```

---

*Parent topic: Inter-Core Synchronization*
