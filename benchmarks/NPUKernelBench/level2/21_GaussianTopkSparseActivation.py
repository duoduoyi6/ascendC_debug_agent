import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Gaussian-based Top-k Sparse Activation Module.

    Computes adaptive sparsity threshold based on input statistics:
    1. Compute mean and std of input across feature dimension
    2. Calculate threshold = mean + std * norm.icdf(target_sparsity)
    3. Apply ReLU(input - threshold) to create sparse activations
    """
    def __init__(self):
        super(Model, self).__init__()
        self.a1 = -3.969683028665376e+01
        self.a2 = 2.209460984245205e+02
        self.a3 = -2.759285104469687e+02
        self.a4 = 1.383577518672690e+02
        self.a5 = -3.066479806614716e+01
        self.a6 = 2.506628277459239e+00
        self.b1 = -5.447609879822406e+01
        self.b2 = 1.615858368580409e+02
        self.b3 = -1.556989798598866e+02
        self.b4 = 6.680131188771972e+01
        self.b5 = -1.328068155288572e+01
        self.c1 = -7.784894002430293e-03
        self.c2 = -3.223964580411365e-01
        self.c3 = -2.400758277161838e+00
        self.c4 = -2.549732539343734e+00
        self.c5 = 4.374664141464968e+00
        self.c6 = 2.938163982698783e+00
        self.d1 = 7.784695709041462e-03
        self.d2 = 3.224671290700398e-01
        self.d3 = 2.445134137142996e+00
        self.d4 = 3.754408661907416e+00

    def _ndtri(self, p: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the standard normal CDF (quantile function).

        Uses Abramowitz and Stegun approximation (formula 26.2.23).
        This is a rational approximation that works well for p in (0, 1).

        Args:
            p: Probability values in range (0, 1)

        Returns:
            Quantile values (z-scores) from standard normal distribution
        """
        p_low = 0.02425
        p_high = 1.0 - p_low
        result = torch.zeros_like(p)
        mask_low = p < p_low
        if mask_low.any():
            q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
            result[mask_low] = (((((self.c1*q + self.c2)*q + self.c3)*q + self.c4)*q + self.c5)*q + self.c6) / ((((self.d1*q + self.d2)*q + self.d3)*q + self.d4)*q + 1.0)
        mask_mid = (p >= p_low) & (p <= p_high)
        if mask_mid.any():
            q = p[mask_mid] - 0.5
            r = q * q
            result[mask_mid] = (((((self.a1*r + self.a2)*r + self.a3)*r + self.a4)*r + self.a5)*r + self.a6)*q / (((((self.b1*r + self.b2)*r + self.b3)*r + self.b4)*r + self.b5)*r + 1.0)
        mask_high = p > p_high
        if mask_high.any():
            q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
            result[mask_high] = -(((((self.c1*q + self.c2)*q + self.c3)*q + self.c4)*q + self.c5)*q + self.c6) / ((((self.d1*q + self.d2)*q + self.d3)*q + self.d4)*q + 1.0)
        return result

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
            target_sparsity: Float in [0, 1] indicating target sparsity level.
                            0.0 means no sparsity (all activations pass through).

        Returns:
            Sparsified tensor of same shape as input.
        """
        if target_sparsity == 0.0:
            return inputs
        input_f32 = inputs.to(torch.float32)
        inputs_mean = torch.mean(input_f32, dim=-1, keepdim=True)
        inputs_std = torch.std(input_f32, dim=-1, keepdim=True, unbiased=False)
        threshold = inputs_mean + inputs_std * self._ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
        output = torch.nn.functional.relu(input_f32 - threshold)
        return output.to(torch.bfloat16)
