#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"

class RmsNormMergeNKernel {
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
            rowLoops_ = subBlockRows_ / tiling_.rowFactor;
            pipe_->InitBuffer(gammaBuf_, AlignedRowBytes());
            pipe_->InitBuffer(xInQueue_, 1, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(yOutQueue_, 1, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(scaleBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(gammaTileBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(gammaBroadcastTmpBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(scaleBroadcastTmpBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(reduceBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));

            gammaLocal_ = gammaBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            const uint32_t gammaSrcShape[2] = {
                1U,
                static_cast<uint32_t>(tiling_.N),
            };
            const uint32_t gammaDstShape[2] = {
                static_cast<uint32_t>(kMergeRowFactor),
                static_cast<uint32_t>(tiling_.N),
            };

            gammaLocal_ = gammaBuf_.Get<float>();
            CopyGmToUbRow(gammaLocal_, gammaGM_);
            AscendC::PipeBarrier<PIPE_MTE2>();

            gammaTileLocal_ = gammaTileBuf_.Get<float>();
            gammaBroadcastTmpLocal_ = gammaBroadcastTmpBuf_.Get<uint8_t>();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Broadcast<float, 2, 0>(
                gammaTileLocal_,
                gammaLocal_,
                gammaDstShape,
                gammaSrcShape,
                gammaBroadcastTmpLocal_);
            AscendC::PipeBarrier<PIPE_V>();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int r = 0; r < rowLoops_; ++r) {
                    const int rowBase = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + r * tiling_.rowFactor;
                    const int validRows = tiling_.M - rowBase;
                    if (validRows > 0) {
                        ProcessRows(rowBase, validRows >= kMergeRowFactor ? kMergeRowFactor : validRows);
                    }
                }
            }
        }
    }

private:
    static constexpr int kMergeRowFactor = 8;

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

    __aicore__ inline void ProcessSingleRow(int rowIdx)
    {
        const uint32_t reduceShape[2] = {
            1U,
            static_cast<uint32_t>(tiling_.N),
        };
        const uint32_t scaleSrcShape[2] = {
            1U,
            1U,
        };
        const uint32_t scaleDstShape[2] = {
            1U,
            static_cast<uint32_t>(tiling_.N),
        };

        xInQueue_.AllocTensor<float>(xLocal_);
        yOutQueue_.AllocTensor<float>(yLocal_);
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        CopyGmToUbRow(xLocal_, xGM_[rowIdx * tiling_.N]);
        xInQueue_.EnQue(xLocal_);

        xInQueue_.DeQue<float>(xLocal_);
        AscendC::Mul(yLocal_, xLocal_, xLocal_, tiling_.N);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_,
            yLocal_,
            reduceTmpLocal_,
            reduceShape,
            true);
        AscendC::Muls(sumLocal_, sumLocal_, tiling_.invN, 1);
        AscendC::Adds(sumLocal_, sumLocal_, tiling_.eps, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_,
            sumLocal_,
            scaleDstShape,
            scaleSrcShape,
            scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, tiling_.N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, tiling_.N);

        xInQueue_.FreeTensor(xLocal_);
        yOutQueue_.EnQue(yLocal_);

        yOutQueue_.DeQue<float>(yLocal_);
        CopyUbToGmRow(yGM_[rowIdx * tiling_.N], yLocal_);
        yOutQueue_.FreeTensor(yLocal_);
    }

    __aicore__ inline void ProcessRows(int rowBase, int validRows)
    {
        if (validRows < kMergeRowFactor) {
            for (int r = 0; r < validRows; ++r) {
                ProcessSingleRow(rowBase + r);
            }
            return;
        }

        const int tileSize = kMergeRowFactor * tiling_.N;
        const uint32_t reduceShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(tiling_.N),
        };
        const uint32_t scaleSrcShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            1U,
        };
        const uint32_t scaleDstShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(tiling_.N),
        };
        xInQueue_.AllocTensor<float>(xLocal_);
        yOutQueue_.AllocTensor<float>(yLocal_);
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        AscendC::DataCopy(xLocal_, xGM_[rowBase * tiling_.N], tileSize);
        xInQueue_.EnQue(xLocal_);

        xInQueue_.DeQue<float>(xLocal_);
        AscendC::Mul(yLocal_, xLocal_, xLocal_, tileSize);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_,
            yLocal_,
            reduceTmpLocal_,
            reduceShape,
            true);
        AscendC::Muls(sumLocal_, sumLocal_, tiling_.invN, kMergeRowFactor);
        AscendC::Adds(sumLocal_, sumLocal_, tiling_.eps, kMergeRowFactor);
        AscendC::Rsqrt(sumLocal_, sumLocal_, kMergeRowFactor);
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_,
            sumLocal_,
            scaleDstShape,
            scaleSrcShape,
            scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, tileSize);
        AscendC::Mul(yLocal_, yLocal_, gammaTileLocal_, tileSize);

        xInQueue_.FreeTensor(xLocal_);
        yOutQueue_.EnQue(yLocal_);

        yOutQueue_.DeQue<float>(yLocal_);
        AscendC::DataCopy(yGM_[rowBase * tiling_.N], yLocal_, tileSize);
        yOutQueue_.FreeTensor(yLocal_);
    }
private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};
    int rowLoops_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> gammaGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;

    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> scaleLocal_;
    AscendC::LocalTensor<float> gammaTileLocal_;
    AscendC::LocalTensor<uint8_t> gammaBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> scaleBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> reduceTmpLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
