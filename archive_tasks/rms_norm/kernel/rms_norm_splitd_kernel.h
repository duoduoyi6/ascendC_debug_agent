#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"

class RmsNormSplitDKernel {
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
            pipe_->InitBuffer(xInQueue_, 1, kTileBytes);
            pipe_->InitBuffer(gammaInQueue_, 1, kTileBytes);
            pipe_->InitBuffer(yOutQueue_, 1, kTileBytes);
            pipe_->InitBuffer(accumBuf_, kTileBytes);
            pipe_->InitBuffer(reduceBuf_, kBlockN * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(tempBuf_, kTileBytes);
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
    static constexpr int kBlockN = 1024;
    static constexpr uint32_t kTileBytes = kBlockN * sizeof(float);

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline int32_t NumTiles() const
    {
        return (tiling_.N + kBlockN - 1) / kBlockN;
    }

    __aicore__ inline int32_t GetValidN(int32_t colBase) const
    {
        return (colBase + kBlockN <= tiling_.N) ? kBlockN : (tiling_.N - colBase);
    }

    __aicore__ inline uint32_t ValidBytes(int32_t validN) const
    {
        return static_cast<uint32_t>(validN * static_cast<int32_t>(sizeof(float)));
    }

    __aicore__ inline void CopyGmToUbTile(
        AscendC::LocalTensor<float> &dst,
        AscendC::GlobalTensor<float> src,
        int32_t validN)
    {
        const uint32_t validBytes = ValidBytes(validN);
        const uint32_t padElems = static_cast<uint32_t>(kBlockN - validN);
        AscendC::DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(padElems), 0.0f};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyUbToGmTile(
        AscendC::GlobalTensor<float> dst,
        AscendC::LocalTensor<float> &src,
        int32_t validN)
    {
        AscendC::DataCopyExtParams copyParams{1, ValidBytes(validN), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        accumLocal_ = accumBuf_.Get<float>();
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        tempLocal_ = tempBuf_.Get<float>();

        AscendC::Duplicate(accumLocal_, 0.0f, kBlockN);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            xInQueue_.AllocTensor<float>(xLocal_);
            CopyGmToUbTile(xLocal_, xGM_[rowIdx * tiling_.N + colBase], validN);
            xInQueue_.EnQue(xLocal_);

            xInQueue_.DeQue<float>(xLocal_);
            AscendC::Mul(tempLocal_, xLocal_, xLocal_, kBlockN);
            AscendC::Add(accumLocal_, accumLocal_, tempLocal_, kBlockN);
            xInQueue_.FreeTensor(xLocal_);
        }

        AscendC::ReduceSum<float>(sumLocal_, accumLocal_, reduceLocal_, kBlockN);
        float invRms = sumLocal_.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, invRms, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);
        invRms = sumLocal_.GetValue(0);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            xInQueue_.AllocTensor<float>(xLocal_);
            gammaInQueue_.AllocTensor<float>(gammaLocal_);
            CopyGmToUbTile(xLocal_, xGM_[rowIdx * tiling_.N + colBase], validN);
            CopyGmToUbTile(gammaLocal_, gammaGM_[colBase], validN);
            xInQueue_.EnQue(xLocal_);
            gammaInQueue_.EnQue(gammaLocal_);

            yOutQueue_.AllocTensor<float>(yLocal_);
            xInQueue_.DeQue<float>(xLocal_);
            gammaInQueue_.DeQue<float>(gammaLocal_);
            AscendC::Muls(yLocal_, xLocal_, invRms, kBlockN);
            AscendC::Mul(yLocal_, yLocal_, gammaLocal_, kBlockN);
            xInQueue_.FreeTensor(xLocal_);
            gammaInQueue_.FreeTensor(gammaLocal_);
            yOutQueue_.EnQue(yLocal_);

            yOutQueue_.DeQue<float>(yLocal_);
            CopyUbToGmTile(yGM_[rowIdx * tiling_.N + colBase], yLocal_, validN);
            yOutQueue_.FreeTensor(yLocal_);
        }
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> gammaGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;

    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> accumLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> tempLocal_;
};
