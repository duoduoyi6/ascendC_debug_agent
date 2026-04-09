#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"

class RmsNormSingleRowKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), tiling_.M * tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(gammaBuf_, AlignedRowBytes());
            pipe_->InitBuffer(xInQueue_, 1, AlignedRowBytes());
            pipe_->InitBuffer(yOutQueue_, 1, AlignedRowBytes());
            pipe_->InitBuffer(reduceBuf_, tiling_.N * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));

            gammaLocal_ = gammaBuf_.Get<float>();
            CopyGmToUbRow(gammaLocal_, gammaGM_);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline uint32_t RowBytes() const
    {
        return static_cast<uint32_t>(tiling_.N * sizeof(float));
    }

    __aicore__ inline uint32_t AlignedRowBytes() const
    {
        const uint32_t rowBytes = RowBytes();
        return ((rowBytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline void CopyGmToUbRow(AscendC::LocalTensor<float> &dst, AscendC::GlobalTensor<float> src)
    {
        const uint32_t padElems = (AlignedRowBytes() - RowBytes()) / sizeof(float);
        AscendC::DataCopyExtParams copyParams{1, RowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(padElems), 0.0f};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyUbToGmRow(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> &src)
    {
        AscendC::DataCopyExtParams copyParams{1, RowBytes(), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        xInQueue_.AllocTensor<float>(xLocal_);
        yOutQueue_.AllocTensor<float>(yLocal_);
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        CopyGmToUbRow(xLocal_, xGM_[rowIdx * tiling_.N]);
        xInQueue_.EnQue(xLocal_);

        xInQueue_.DeQue<float>(xLocal_);
        AscendC::Mul(yLocal_, xLocal_, xLocal_, tiling_.N);
        AscendC::ReduceSum<float>(sumLocal_, yLocal_, reduceLocal_, tiling_.N);

        float meanSq = sumLocal_.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, meanSq, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);

        float invRms = sumLocal_.GetValue(0);
        AscendC::Muls(yLocal_, xLocal_, invRms, tiling_.N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, tiling_.N);

        xInQueue_.FreeTensor(xLocal_);
        yOutQueue_.EnQue(yLocal_);

        yOutQueue_.DeQue<float>(yLocal_);
        CopyUbToGmRow(yGM_[rowIdx * tiling_.N], yLocal_);
        yOutQueue_.FreeTensor(yLocal_);
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> gammaGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;

    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
